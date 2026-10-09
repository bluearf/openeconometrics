"""Option-level independent oracles, persistence, streaming and failure gates."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize, minimize_scalar

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import extraction, rotation


def data(seed=413, n=180):
    rng = np.random.default_rng(seed)
    loadings = np.array([[.8, .1], [.7, .05], [.75, .2], [.1, .8], [.05, .7], [.2, .75]])
    x = rng.normal(size=(n, 2)) @ loadings.T + .55 * rng.normal(size=(n, 6))
    return pd.DataFrame(x * np.arange(1, 7) + np.arange(10, 16), columns=list("abcdef"))


def stream(frame):
    def batches():
        for start in range(0, len(frame), 17):
            yield frame.iloc[start:start+17]
    return Dataset.from_batches(batches, columns=list(frame.columns), row_count=len(frame))


def saved(result):
    payload = json.loads(json.dumps({"attrs": result.attrs, "tables": {
        key: json.loads(value.to_json(orient="split", double_precision=15))
        for key, value in result.items()}}, allow_nan=False))
    return TableSet({key: pd.DataFrame(**table) for key, table in payload["tables"].items()}, **payload["attrs"])


def equal_tables(one, two, atol=1e-9):
    assert set(one) == set(two)
    for key in one:
        np.testing.assert_allclose(one[key], two[key], atol=atol, rtol=atol, equal_nan=True)


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
@pytest.mark.parametrize("streaming", [False, True])
def test_frequency_replication_and_saved_scores(matrix, streaming):
    frame = data(n=80)
    frame["w"] = np.arange(len(frame)) % 5
    frame.loc[9, "a"] = np.nan
    frame.loc[12, "w"] = np.nan
    clean = frame.dropna()
    repeated = clean.loc[clean.index.repeat(clean.w.astype(int))]
    fit = oe.pca(stream(frame) if streaming else frame, list("abcdef"), weights="w", matrix=matrix, components=3)
    reference = oe.pca(repeated, list("abcdef"), matrix=matrix, components=3)
    equal_tables(fit, reference)
    assert fit.attrs["n"] == int(clean.w.sum())
    assert fit.attrs["physical_rows"] == int((clean.w > 0).sum())
    assert fit.attrs["n_missing"] == 2
    assert fit.attrs["n_zero_weight"] == int((clean.w == 0).sum())
    np.testing.assert_allclose(oe.pca_scores(saved(fit), frame), oe.pca_scores(reference, frame), atol=1e-10, equal_nan=True)
    if streaming:
        assert fit.attrs["source_content_sha256"] and fit.attrs["source_passes"] == 1
        scores = oe.pca_scores(saved(fit), stream(frame))
        blocks = list(scores.iter_batches())
        np.testing.assert_allclose(pd.concat(blocks), oe.pca_scores(fit, frame), atol=1e-10, equal_nan=True)


def test_frequency_large_location_and_zero_weights():
    frame = data(n=50) + 1e10
    frame["w"] = 3
    frame.loc[0, "w"] = 0
    fit = oe.pca(stream(frame), list("abcdef"), weights="w", matrix="covariance")
    replicated = frame.loc[frame.index.repeat(frame.w)]
    expected = np.cov(replicated[list("abcdef")].to_numpy().T)
    np.testing.assert_allclose(fit["eigenvalues"].eigenvalue, np.linalg.eigvalsh(expected)[::-1], rtol=1e-7)


@pytest.mark.parametrize("bad", [-1, .5, np.inf, 2**54])
def test_bad_frequency_weights(bad):
    frame = data(n=30).assign(w=1.)
    frame.loc[0, "w"] = bad
    with pytest.raises(AnalysisError):
        oe.pca(stream(frame), list("abcdef"), weights="w")


def test_frequency_refuses_other_weights_and_precision_overflow():
    frame = data(n=3).assign(w=2**52)
    with pytest.raises(AnalysisError, match="precision"):
        oe.pca(frame, list("abcdef"), weights="w")
    with pytest.raises(AnalysisError):
        oe.pca(frame, list("abcdef"), weights="w", weight_type="aweight")
    with pytest.raises(AnalysisError):
        oe.pca(frame.assign(w=0), list("abcdef"), weights="w")
    integer_edge = data(n=3).assign(w=[2**53+1, 0, 0])
    with pytest.raises(AnalysisError, match="precision"):
        oe.pca(integer_edge, list("abcdef"), weights="w")


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_summary_pca_raw_matrix_persistence(matrix):
    frame = data()
    summary = frame.cov() if matrix == "covariance" else frame.corr()
    fit = oe.pca_matrix(summary, n=len(frame), matrix=matrix, means=frame.mean(), sds=frame.std(), components=2)
    raw = oe.pca(frame, list(frame), matrix=matrix, components=2)
    equal_tables(fit, raw)
    np.testing.assert_allclose(oe.pca_scores(saved(fit), frame), oe.pca_scores(raw, frame), atol=1e-10)
    assert fit.attrs["input_kind"] == "summary_matrix"
    assert fit.attrs["n_missing"] == 0


@pytest.mark.parametrize("method", ["pf", "ipf", "pcf", "ml", "minres"])
@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_summary_factor_raw_equivalence(method, matrix):
    frame = data()
    summary = frame.cov() if matrix == "covariance" else frame.corr()
    options = {"method": method, "factors": 2, "rotate": "varimax", "max_iterations": 1000}
    fit = oe.factor_matrix(summary, n=len(frame), matrix=matrix, means=frame.mean(), sds=frame.std(), **options)
    raw = oe.factor(frame, list(frame), **options)
    equal_tables(fit, raw, atol=2e-7)
    np.testing.assert_allclose(oe.factor_scores(saved(fit), frame), oe.factor_scores(raw, frame), atol=2e-7)


def test_summary_rank_deficient_pca_and_moment_gates():
    fit = oe.pca_matrix([[1, 1], [1, 1]], n=40, columns=["a", "b"])
    assert fit.attrs["components"] == 1
    with pytest.raises(AnalysisError, match="training means"):
        oe.pca_scores(saved(fit), {"a": [1], "b": [2]})
    means_only = oe.pca_matrix([[1, .3], [.3, 1]], n=40, columns=["a", "b"], means=[0, 0])
    with pytest.raises(AnalysisError, match="standard deviations"):
        oe.pca_scores(means_only, {"a": [1], "b": [2]})
    raw_only = oe.pca_matrix([[4, .3], [.3, 1]], n=40, columns=["a", "b"], matrix="covariance")
    assert oe.pca_scores(raw_only, {"a": [1], "b": [2]}, center=False).notna().all().all()
    factor = oe.factor_matrix(data().corr(), n=180, factors=2)
    with pytest.raises(AnalysisError, match="training means"):
        oe.factor_scores(factor, data())


@pytest.mark.parametrize("values", [ [[1, .2], [.3, 1]], [[1, 2], [2, 1]], [[1, .2], [.2, 0]],
                                    [[1, np.inf], [np.inf, 1]], [[2, 0], [0, 1]], [[1, 0]] ])
def test_bad_summary(values):
    with pytest.raises(AnalysisError):
        oe.pca_matrix(values, n=40, columns=["a", "b"])


def test_summary_label_and_scale_refusal():
    frame = data()
    with pytest.raises(AnalysisError):
        oe.pca_matrix(frame.cov().iloc[::-1], n=180)
    with pytest.raises(AnalysisError):
        oe.pca_matrix(frame.cov(), n=180, matrix="covariance", sds=[1]*6)
    with pytest.raises(AnalysisError):
        oe.factor_matrix(frame.corr(), n=2, factors=2)
    with pytest.raises(AnalysisError):
        oe.pca_matrix(np.eye(2, dtype=complex), n=40, columns=["a", "b"])
    with pytest.raises(AnalysisError):
        oe.pca_matrix(torch.eye(2, dtype=torch.complex128), n=40, columns=["a", "b"])
    with pytest.raises(AnalysisError):
        oe.factor_matrix(frame.corr(), n=40, factors=2, max_iterations="many")


@pytest.mark.parametrize("seed", [1, 21, 301])
def test_minres_independent_scipy_objective_and_stationarity(seed):
    frame = data(seed=seed, n=250)
    r = frame.corr().to_numpy()
    fit = oe.factor(frame, list(frame), method="minres", factors=2)
    ours = fit["loadings"].to_numpy()
    # Independently differentiated NumPy residual objective, starting elsewhere.
    def objective(theta):
        a = theta.reshape(6, 2)
        residual = a @ a.T - r
        np.fill_diagonal(residual, 0)
        return .5 * (residual**2).sum(), (2 * residual @ a).ravel()
    eig, vectors = np.linalg.eigh(r)
    start = vectors[:, -2:] * np.sqrt(eig[-2:])
    ref = minimize(objective, start.ravel(), jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 2000})
    np.testing.assert_allclose(ours @ ours.T, ref.x.reshape(6, 2) @ ref.x.reshape(6, 2).T, atol=2e-6)
    assert abs(fit.attrs["discrepancy"] - ref.fun) < 1e-9
    assert fit.attrs["gradient_max"] < 1e-7
    assert "model" not in fit["fit"].index
    assert fit.attrs["converged"]
    np.testing.assert_allclose(fit["uniqueness"].uniqueness, 1 - (ours**2).sum(1), atol=1e-12)
    equal_tables(fit, oe.factor(stream(frame), list(frame), method="minres", factors=2), atol=1e-6)


def test_minres_derivative_and_failures():
    frame = data()
    r = torch.tensor(frame.corr().to_numpy())
    a = torch.tensor(np.random.default_rng(1).normal(size=(6, 2)) * .2)
    _, gradient = extraction.minres_objective(a, r)
    for i in range(6):
        for j in range(2):
            plus, minus = a.clone(), a.clone()
            plus[i, j] += 1e-6
            minus[i, j] -= 1e-6
            numeric = (extraction.minres_objective(plus, r)[0] - extraction.minres_objective(minus, r)[0]) / 2e-6
            assert float(gradient[i, j]) == pytest.approx(float(numeric), abs=1e-9)
    for options in ({}, {"factors": 5}, {"factors": 2, "max_iterations": 1}, {"factors": 2, "mineigen": 0}):
        with pytest.raises(AnalysisError):
            oe.factor(frame, list(frame), method="minres", **options)


@pytest.mark.parametrize("rotate", [None, "varimax", "geomin"])
def test_anderson_rubin_unit_covariance_and_independent_root(rotate):
    frame = data()
    fit = oe.factor(frame, list(frame), factors=2, rotate=rotate, scores="anderson_rubin")
    pattern = fit["rotated_loadings" if rotate else "loadings"].to_numpy()
    weighted = pattern / fit["uniqueness"].to_numpy()
    covariance = weighted.T @ frame.corr().to_numpy() @ weighted
    values, vectors = np.linalg.eigh(covariance)
    expected = weighted @ ((vectors / np.sqrt(values)) @ vectors.T)
    np.testing.assert_allclose(fit["score_coefficients"], expected, atol=1e-10)
    scores = oe.factor_scores(saved(fit), frame)
    np.testing.assert_allclose(np.cov(scores.to_numpy().T), np.eye(2), atol=1e-10)
    holes = frame.copy()
    holes.index = np.arange(180) * 3
    holes.iloc[3, 2] = np.nan
    assert oe.factor_scores(saved(fit), holes).iloc[3].isna().all()
    np.testing.assert_allclose(pd.concat(list(oe.factor_scores(saved(fit), stream(frame)).iter_batches())), scores, atol=1e-12)


@pytest.mark.parametrize("rotate,extras", [("promax", {}), ("geomin", {"geomin_oblique": True})])
def test_anderson_rubin_refuses_oblique(rotate, extras):
    with pytest.raises(AnalysisError, match="orthogonal"):
        oe.factor(data(), list("abcdef"), factors=2, rotate=rotate, scores="anderson_rubin", **extras)


@pytest.mark.parametrize("kaiser", [True, False])
def test_target_global_svd_orientation_and_covariance(kaiser):
    frame = data()
    unrotated = oe.factor(frame, list(frame), factors=2)["loadings"].to_numpy()
    target = unrotated @ np.array([[0., -1.], [1., 0.]])
    fit = oe.factor(frame, list(frame), factors=2, rotate="target", target=target, kaiser=kaiser)
    np.testing.assert_allclose(fit["rotated_loadings"], target, atol=1e-12)
    matrix = fit["rotation_matrix"].to_numpy()
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(fit["rotated_loadings"].to_numpy() @ fit["rotated_loadings"].to_numpy().T, unrotated @ unrotated.T, atol=1e-12)
    assert fit.attrs["rotation_orientation"] == "caller target order/sign"
    np.testing.assert_allclose(saved(fit)["rotation_target"], target, atol=1e-14)
    # Non-exact target: independent NumPy Procrustes formula.
    target = target + np.random.default_rng(1).normal(size=(6, 2)) * .1
    scale = np.sqrt((unrotated**2).sum(1)) if kaiser else np.ones(6)
    u, _, vt = np.linalg.svd((unrotated / scale[:, None]).T @ (target / scale[:, None]))
    fit = oe.factor(frame, list(frame), factors=2, rotate="target", target=target, kaiser=kaiser)
    np.testing.assert_allclose(fit["rotation_matrix"], u @ vt, atol=1e-12)


@pytest.mark.parametrize("target", [None, [[1, 0]], np.zeros((6, 2)), np.full((6, 2), np.nan)])
def test_target_refuses_bad_geometry(target):
    with pytest.raises(AnalysisError):
        oe.factor(data(), list("abcdef"), factors=2, rotate="target", target=target)


@pytest.mark.parametrize("oblique", [False, True])
def test_geomin_criterion_gradient_and_independent_optimization(oblique):
    frame = data()
    a = oe.factor(frame, list(frame), factors=2)["loadings"].to_numpy()
    epsilon = .01
    fit = oe.factor(frame, list(frame), factors=2, rotate="geomin", kaiser=False, geomin_oblique=oblique)
    pattern = fit["rotated_loadings"].to_numpy()
    def criterion(z):
        return np.sqrt(np.prod(z**2 + epsilon, axis=1)).sum()
    if oblique:
        def objective(angles):
            t = np.array([[np.cos(angles[0]), -np.sin(angles[1])], [np.sin(angles[0]), np.cos(angles[1])]])
            return criterion(a @ np.linalg.inv(t).T)
        ref = minimize(objective, [.0, .0], method="BFGS", options={"gtol": 1e-9})
        phi = fit["factor_correlations"].to_numpy()
        np.testing.assert_allclose(np.diag(phi), 1, atol=1e-12)
    else:
        def objective(theta):
            return criterion(a @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))
        # Search the full 2-factor orthogonal criterion rather than reuse our start.
        candidates = [minimize_scalar(objective, bounds=(v, v + np.pi / 16), method="bounded") for v in np.arange(0, np.pi, np.pi / 16)]
        ref = min(candidates, key=lambda item: item.fun)
        phi = np.eye(2)
    assert abs(criterion(pattern) - ref.fun) < 1e-8
    np.testing.assert_allclose(pattern @ phi @ pattern.T, a @ a.T, atol=1e-11)
    assert fit.attrs["rotation_converged"]
    values = torch.tensor(a)
    _, gradient = rotation.geomin_value(values, epsilon)
    for i in range(6):
        for j in range(2):
            plus, minus = a.copy(), a.copy()
            plus[i, j] += 1e-6
            minus[i, j] -= 1e-6
            assert float(gradient[i, j]) == pytest.approx((criterion(plus) - criterion(minus)) / 2e-6, abs=1e-9)


def test_geomin_failures_and_workspace_gate(monkeypatch):
    for options in ({"max_iterations": 1}, {"geomin_epsilon": 0}, {"geomin_oblique": 1}):
        with pytest.raises(AnalysisError):
            oe.factor(data(), list("abcdef"), factors=2, rotate="geomin", **options)
    from openecon.resources import use_workspace_budget
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        oe.pca_matrix(np.eye(60), n=100, columns=[f"x{i}" for i in range(60)])
    with pytest.raises(AnalysisError, match="work budget"):
        oe.factor_matrix(np.eye(50), n=100, columns=[f"x{i}" for i in range(50)], factors=2, rotate="geomin", max_iterations=100000)


def correspondence():
    counts = np.array([[35, 12, 3], [8, 31, 10], [3, 8, 26], [12, 5, 4]])
    frame = pd.DataFrame([(f"r{i}", f"c{j}", int(counts[i,j])) for i in range(4) for j in range(3)], columns=["r", "c", "n"])
    return oe.ca(frame, "r", "c", weights="n"), counts


@pytest.mark.parametrize("axis", ["row", "column"])
def test_supplementary_barycentric_projection_persistence_and_invariance(axis):
    fit, counts = correspondence()
    before = json.dumps(fit.attrs, sort_keys=True)
    opposite = "column" if axis == "row" else "row"
    active = counts if axis == "row" else counts.T
    labels = fit.attrs["projection_state"][f"{opposite}_labels"]
    points = pd.DataFrame(active, columns=labels, index=[f"s{i}" for i in range(len(active))])
    projected = oe.ca_project(saved(fit), points[labels[::-1]], axis=axis)
    ref_table = fit["rows" if axis == "row" else "columns"]
    np.testing.assert_allclose(projected[["dim1", "dim2"]], ref_table[["dim1", "dim2"]], atol=1e-12)
    profile = active / active.sum(axis=1, keepdims=True)
    standard = np.array(fit.attrs["projection_state"][f"{opposite}_standard"])
    np.testing.assert_allclose(projected[["dim1", "dim2"]], profile @ standard, atol=1e-12)
    assert json.dumps(fit.attrs, sort_keys=True) == before
    assert list(projected.index) == list(points.index)
    assert projected.attrs["supplementary"]


@pytest.mark.parametrize("profiles", [{"c0": [1], "c1": [2]}, {"c0": [0], "c1": [0], "c2": [0]},
                                      {"c0": [-1], "c1": [2], "c2": [3]}, {"c0": [1], "c1": [np.nan], "c2": [3]}])
def test_supplementary_bad_profiles(profiles):
    fit, _ = correspondence()
    with pytest.raises(AnalysisError):
        oe.ca_project(fit, profiles)


def test_supplementary_zero_inertia_axes_and_corruption():
    counts = np.array([[20, 10, 10], [10, 20, 20], [15, 15, 15]])
    frame = pd.DataFrame([(f"r{i}", f"c{j}", int(counts[i,j])) for i in range(3) for j in range(3)], columns=["r", "c", "n"])
    fit = oe.ca(frame, "r", "c", weights="n")
    result = oe.ca_project(fit, {"c0": [2], "c1": [1], "c2": [1]})
    assert result.dim2_standard.isna().all()
    null_direction = oe.ca_project(fit, {"c0": [1], "c1": [2], "c2": [8]})
    assert null_direction[["dim2", "dim2_standard", "dim2_sqcorr"]].isna().all().all()
    centroid = oe.ca_project(fit, {"c0": [45], "c1": [45], "c2": [45]})
    assert float(centroid.quality.iloc[0]) == 0
    assert float(centroid.dim1.iloc[0]) == 0
    bad = saved(fit)
    bad.attrs["projection_state"]["column_mass"][0] = 0
    with pytest.raises(AnalysisError, match="state"):
        oe.ca_project(bad, {"c0": [1], "c1": [1], "c2": [1]})


def test_public_registry_and_latex():
    from openecon.econometrics import registry
    for name in ("pca_matrix", "factor_matrix", "ca_project"):
        assert name in registry.public_exports()
    fit = oe.factor(data(), list("abcdef"), method="minres", factors=2, rotate="geomin", scores="anderson_rubin")
    assert fit.to_latex().count(r"\begin{tabular}") == len(fit)
    json.dumps(fit.attrs, allow_nan=False)
