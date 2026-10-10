"""Independent NumPy/SciPy refits of all eight fixed-query score domains.

The reference uses literal expansion, rectangular SVD, covariance-pencil CCA
and spectral PF/Procrustes. Only integer RNG ranks are shared with production.
"""
from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import score_uncertainty as u
from openecon.econometrics.multivariate import frequency_bootstrap as f
from openecon.econometrics.multivariate import canon_uncertainty as cu
from openecon.econometrics.multivariate import factor_uncertainty as fu
from openecon.econometrics.postest.index_codec import decode
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget

import test_pca_frequency_uncertainty as pr
import test_canon_frequency_uncertainty as cr
import test_factor_frequency_uncertainty as fr

DOMAINS = ("pca_covariance", "pca_correlation", "cca_iid", "cca_frequency",
           "pf_iid", "target_iid", "pf_frequency", "target_frequency")


def setup(domain, distribution="normal", *, shift=0., scale=1., b=39, missing=False, factors=2):
    if domain.startswith("pca"):
        frame = pr.domain("gaussian" if distribution == "normal" else "skewed_t8")
        names, weight, target = pr.NAMES, "freq", None
    elif domain.startswith("cca"):
        frame = cr.fixture(distribution)
        names, weight, target = cr.NAMES, "frequency", None
    else:
        frame, target = fr.fixture(distribution, factors=factors)
        names, weight = list(frame.drop(columns="frequency")), "frequency"
        target = target if domain.startswith("target") else None
    frame[names] = frame[names] * scale + shift
    if missing:
        frame.index = pd.MultiIndex.from_tuples([("unit", i % 7) for i in range(len(frame))], names=["block", "id"])
        frame.columns.name = "measurements"
        frame.iloc[1, 0] = np.nan
        frame.loc[frame.index == ("unit", 4), weight] = np.nan
    options = dict(replications=b, confidence=.9, seed=19)
    if domain.startswith("pca"):
        fit = oe.pca_fweight_bootstrap(frame, names, weights=weight, components=2,
                                      matrix=domain.removeprefix("pca_"), **options)
    elif domain.startswith("cca"):
        function = oe.canon_fweight_bootstrap if "frequency" in domain else oe.canon_bootstrap
        extra = {"weights": weight} if "frequency" in domain else {}
        fit = function(frame, cr.X, cr.Y, components=2, target="coefficients", anchors=cr.X[:2], **extra, **options)
    else:
        function = oe.factor_multifactor_fweight_bootstrap if "frequency" in domain else oe.factor_multifactor_bootstrap
        extra = {"weights": weight} if "frequency" in domain else {}
        fit = function(frame, names, factors=factors, target=target, **extra, **options)
    # Independent fixed queries, generated from a different RNG before use.
    query = pd.DataFrame(np.random.default_rng(890).normal(size=(3, len(names))) * scale + shift,
                         columns=names)
    return frame, names, weight, target, fit, query


@lru_cache(maxsize=32)
def cached(domain, distribution="normal"):
    return setup(domain, distribution)


def score(domain, fit, query, **options):
    function = oe.pca_fweight_bootstrap_scores if domain.startswith("pca") else \
        oe.canon_bootstrap_scores if domain.startswith("cca") else oe.factor_bootstrap_scores
    return function(fit, query, **options)


def reference(domain, values, query, fit, target):
    # Stable independently centered represented values, including large shifts.
    origin = values[0]
    offset = (values - origin).mean(0)
    centered = (values - origin) - offset
    correction = centered.mean(0)
    centered -= correction
    z = (query - origin) - (offset + correction)
    covariance = centered.T @ centered / (len(values) - 1)
    sd = np.sqrt(covariance.diagonal())
    if domain.startswith("pca"):
        result = pr.oracle(values, matrix=fit.attrs["matrix"], components=2, subspace=False,
                           anchors=[fit.attrs["variables"].index(a) for a in fit.attrs["sign_anchors"]])
        coefficients = result["vectors"]
        if domain == "pca_correlation":
            z /= sd
    elif domain.startswith("cca"):
        result = cr.oracle(values, 2, "coefficients")
        coefficients = np.zeros((6, 4))
        coefficients[:3, :2] = result["matrices"]["x_coefficients"]
        coefficients[3:, 2:] = result["matrices"]["y_coefficients"]
    else:
        anchors = None if target is not None else [fit.attrs["variables"].index(a) for a in fit.attrs["sign_anchors"]]
        result = fr.oracle(values, fit.attrs["factors"], anchors, target)
        coefficients = np.linalg.solve(result["correlation"], result["loadings"])
        z /= sd
    return (z @ coefficients).ravel(), coefficients


def oracle(domain, frame, names, weight, target, fit, query):
    if "frequency" in domain or domain.startswith("pca"):
        source = frame[names + [weight]].dropna()
        values = np.repeat(source[names].to_numpy(), source[weight].to_numpy(dtype=int), axis=0)
    else:
        values = frame[names].dropna().to_numpy()
    point, coefficient = reference(domain, values, query.to_numpy(), fit, target)
    generator = torch.Generator(device="cpu").manual_seed(fit.attrs["seed"])
    draws, coefficients = [], []
    for _ in range(fit.attrs["replications"]):
        indices = torch.randint(len(values), (len(values),), generator=generator).numpy()
        draw, coef = reference(domain, values[indices], query.to_numpy(), fit, target)
        draws.append(draw)
        coefficients.append(coef.ravel())
    return point, np.array(draws), coefficient, np.array(coefficients)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("distribution", ["normal", "skew_t8"])
def test_eight_domains_complete_independent_refits_joint_inference(domain, distribution):
    frame, names, weight, target, fit, query = cached(domain, distribution)
    before = summary_state(fit)
    output = score(domain, fit, query)
    point, draws, coefficient, coefficients = oracle(domain, frame, names, weight, target, fit, query)
    np.testing.assert_allclose(output["point_scores"].to_numpy().ravel(), point, atol=3e-11)
    np.testing.assert_allclose(output["replicates"], draws, atol=3e-11)
    np.testing.assert_allclose(output["point_score_coefficients"], coefficient, atol=3e-11)
    np.testing.assert_allclose(output["replicate_score_coefficients"], coefficients, atol=3e-11)
    covariance = np.cov(draws, rowvar=False, ddof=1)
    np.testing.assert_allclose(output["covariance"], covariance, atol=3e-11)
    np.testing.assert_allclose(output["estimates"].std_error, np.sqrt(covariance.diagonal()), atol=3e-11)
    np.testing.assert_allclose(output["estimates"][["ci_lower", "ci_upper"]], np.quantile(draws, [.05, .95], axis=0).T, atol=3e-11)
    np.testing.assert_allclose(output["estimates"].bootstrap_bias, draws.mean(0) - point, atol=3e-11)
    assert np.max(np.abs(covariance[:2, -2:])) > 1e-5
    assert output["estimates"][["p_value", "df"]].isna().all().all()
    assert not output.attrs["familywise_intervals"]
    assert output.attrs["covariance_divisor"] == 38
    assert output.attrs["saved_fit_validation"] == "complete canonical point and every seeded bootstrap refit"
    assert summary_state(fit) == before


@pytest.mark.parametrize("domain", DOMAINS)
def test_large_location_stable_functional_and_source_replay(domain):
    frame, names, weight, target, fit, query = setup(domain, shift=1e12)
    out = score(domain, fit, query)
    point, draws, _, _ = oracle(domain, frame, names, weight, target, fit, query)
    np.testing.assert_allclose(out["point_scores"].to_numpy().ravel(), point, atol=5e-11)
    np.testing.assert_allclose(out["replicates"], draws, atol=5e-11)


@pytest.mark.parametrize("domain", DOMAINS)
def test_source_and_complete_output_roundtrip_missing_and_typed_identities(domain):
    _, names, _, _, fit, query = setup(domain, missing=True)
    query.index = pd.Index([("duplicate", 1), ("duplicate", 1), pd.Timestamp("2026-01-04", tz="UTC")], tupleize_cols=False, name="query")
    query.columns.name = "measurements"
    query.iloc[1, 0] = np.nan
    restored = restore_summary(summary_state(fit))
    out = score(domain, restored, query)
    assert out.attrs["query_positions"] == [0, 2]
    assert out["point_scores"].iloc[1].isna().all()
    assert out["query"].iloc[1].isna().all()
    assert [decode(code) for code in out.attrs["query_index_codes"]] == query.index.tolist()
    assert out.attrs["parameter_order"] == [[i, axis] for i in (0, 2) for axis in out.attrs["axes"]]
    assert summary_state(restore_summary(summary_state(out))) == summary_state(out)
    for name in fit:
        pd.testing.assert_frame_equal(out["fit__" + name], restored[name])
    with pytest.raises(AnalysisError):
        score(domain, fit, query, missing="raise")


@pytest.mark.parametrize("domain", DOMAINS)
def test_positive_unit_invariance_or_covariance_score_scaling(domain):
    *_, base, query = cached(domain)
    *_, transformed, transformed_query = setup(domain, scale=3., shift=7.)
    first, other = score(domain, base, query), score(domain, transformed, transformed_query)
    multiplier = 3 if domain == "pca_covariance" else 1
    np.testing.assert_allclose(other["replicates"], first["replicates"] * multiplier, atol=3e-11)
    np.testing.assert_allclose(other["covariance"], first["covariance"] * multiplier**2, atol=3e-11)


@pytest.mark.parametrize("domain", DOMAINS)
def test_repeated_query_full_singular_covariance_preserved(domain):
    *_, fit, query = cached(domain)
    out = score(domain, fit, pd.concat([query.iloc[:1]] * 2))
    k = len(out.attrs["axes"])
    covariance = out["covariance"].to_numpy()
    np.testing.assert_allclose(covariance[:k, :k], covariance[:k, k:], atol=1e-14)
    assert np.linalg.matrix_rank(covariance) <= k


def reseal(fit):
    if "fweight" in fit.attrs["procedure"]:
        f.seal(fit)
    else:
        digest = cu._state_digest if fit.attrs["procedure"] == "canon_bootstrap" else fu._state_digest
        fit.attrs["state_content_sha256"] = digest(fit)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("mutation", ["draw", "sample", "moment", "seed", "coefficient"])
def test_resealed_semantic_forgery_never_reused(domain, mutation):
    *_, original, query = cached(domain)
    fit = restore_summary(summary_state(original))
    if mutation == "seed":
        fit.attrs["seed"] += 1
    else:
        key = "replicates" if mutation == "draw" else "sample" if mutation == "sample" else \
            "descriptives" if mutation == "moment" else "point_eigenvectors" if domain.startswith("pca") else \
            "point_x_coefficients" if domain.startswith("cca") else "point_loadings"
        fit[key].iloc[0, 0] += .001
    reseal(fit)
    with pytest.raises(AnalysisError, match="canonical|replay"):
        score(domain, fit, query)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("mutation", ["extra", "shape", "metadata", "boolean", "work"])
def test_invalid_saved_admission_precedes_serialization_or_replay(monkeypatch, domain, mutation):
    *_, original, query = cached(domain)
    fit = copy.deepcopy(original)
    if mutation == "extra":
        fit["extra"] = pd.DataFrame([[1.]])
    elif mutation == "shape":
        fit["replicates"] = pd.concat([fit["replicates"]] * 2)
    elif mutation == "metadata":
        fit.attrs["extra"] = "x" * (4 * 1024**2 + 1)
    elif mutation == "boolean":
        fit["replicates"] = fit["replicates"].astype(bool)
    else:
        fit.attrs["estimated_work"] = True
    def forbidden(*args, **kwargs):
        pytest.fail("Untrusted state reached serialization or numerical replay")
    monkeypatch.setattr(u, "summary_state", forbidden)
    monkeypatch.setattr(u, "_replay", forbidden)
    with pytest.raises(AnalysisError):
        score(domain, fit, query)


@pytest.mark.parametrize("domain", DOMAINS)
def test_budget_and_joint_geometry_precede_conversion(monkeypatch, domain):
    *_, fit, query = cached(domain)
    def forbidden(*args, **kwargs):
        pytest.fail("Budget rejection reached source conversion/replay")
    monkeypatch.setattr(u, "_replay", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        score(domain, fit, query)
    with pytest.raises(AnalysisError):
        score(domain, fit, pd.concat([query] * 44))
    monkeypatch.setattr(u, "MAX_WORK", 1)
    with pytest.raises(AnalysisError):
        score(domain, fit, query)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("kind", ["boolean", "infinite", "empty", "all_missing", "columns", "unsupported"])
def test_invalid_query_never_returns_partial_interval(domain, kind):
    *_, fit, original = cached(domain)
    query = original.copy()
    if kind == "boolean":
        query[query.columns[0]] = pd.Series([True] * len(query), index=query.index, dtype=object)
    elif kind == "infinite":
        query.iloc[0, 0] = np.inf
    elif kind == "empty":
        query = query.iloc[:0]
    elif kind == "all_missing":
        query[:] = np.nan
    elif kind == "columns":
        query = query.iloc[:, :-1]
    else:
        query = object()
    with pytest.raises(AnalysisError):
        score(domain, fit, query)


@pytest.mark.parametrize("factors", [3, 4])
@pytest.mark.parametrize("domain", ["pf_iid", "target_iid", "pf_frequency", "target_frequency"])
def test_three_four_factor_complete_functional(domain, factors):
    frame, names, weight, target, fit, query = setup(domain, factors=factors)
    out = score(domain, fit, query)
    point, draws, _, _ = oracle(domain, frame, names, weight, target, fit, query)
    np.testing.assert_allclose(out["point_scores"].to_numpy().ravel(), point, atol=4e-11)
    np.testing.assert_allclose(out["replicates"], draws, atol=4e-11)


def test_correlation_root_only_state_does_not_substitute_coefficients():
    source = cr.fixture()
    fit = oe.canon_bootstrap(source, cr.X, cr.Y, components=2, target="correlations", replications=39, confidence=.9)
    with pytest.raises(AnalysisError, match="coefficient target"):
        oe.canon_bootstrap_scores(fit, source.iloc[:2])
