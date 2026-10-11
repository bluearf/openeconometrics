"""Independent concentrated-GLS and generalized-image algorithm oracles.

The IBM FACTOR algorithms (pp. 3--6 and 8--9) define the objectives being
tested. SciPy optimization uses finite differences and never calls native
factor extraction. Generalized eigenvectors use scipy.linalg.eigh(R, D), not
the production whitening/eigendecomposition implementation.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import linalg, optimize

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import extraction as ex
from openecon.econometrics.multivariate.extraction_additional import (
    generalized_image_factors, generalized_least_squares, gls_discrepancy,
)
from openecon.engines.contracts import KernelError


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def correlation(seed=410, variables=6):
    rng = np.random.default_rng(seed)
    loading = np.array([[.8, .12], [.7, .21], [.72, .08],
                        [.11, .82], [.18, .76], [.08, .7]])
    if variables != 6:
        loading = rng.uniform(.15, .6, (variables, 2))
    data = rng.normal(size=(471, 2)) @ loading.T + rng.normal(size=(471, variables)) * .55
    return np.corrcoef(data, rowvar=False)


def native_gls(r, factors=2, max_iter=1000, **options):
    smc = 1 - 1 / np.linalg.inv(r).diagonal()
    return generalized_least_squares(tensor(r), tensor(smc), factors,
                                     max_iter=max_iter, **options)


def scipy_gls(r, factors):
    """Independent finite-difference constrained optimization in uniqueness units."""
    def objective(psi):
        diagonal = np.diag(1 / np.sqrt(psi))
        roots = np.linalg.eigvalsh(diagonal @ r @ diagonal)
        return np.sum(np.square(roots[:-factors] - 1)) / 2

    result = optimize.minimize(objective, np.full(len(r), .45), method="Nelder-Mead",
                               bounds=[(.005, None)] * len(r),
                               options={"xatol": 2e-11, "fatol": 2e-14, "maxiter": 10000})
    assert result.success and result.status == 0, result.message
    return result.x, objective(result.x)


@pytest.mark.parametrize("seed", [410, 581])
def test_gls_complete_fit_matches_independent_optimizer(seed):
    r = correlation(seed)
    fitted = native_gls(r)
    psi, discrepancy = scipy_gls(r, 2)
    assert_allclose(fitted.uniqueness, psi, rtol=2e-6, atol=2e-7)
    assert fitted.discrepancy == pytest.approx(discrepancy, abs=3e-13)
    diagonal = np.diag(np.sqrt(psi))
    roots, vectors = np.linalg.eigh(np.linalg.solve(diagonal, r) @ np.linalg.inv(diagonal))
    reference = diagonal @ vectors[:, -2:] @ np.diag(np.sqrt(roots[-2:] - 1))
    assert_allclose(fitted.loadings @ fitted.loadings.T, reference @ reference.T, atol=3e-7)
    reproduced = reference @ reference.T + np.diag(psi)
    assert_allclose(fitted.matrices["reproduced_correlation"], reproduced, atol=3e-7)
    assert_allclose(fitted.matrices["optimized_uniqueness"], np.diag(psi), atol=2e-7)
    # The complete fit discrepancy is also a sample-versus-model generalized
    # eigenvalue discrepancy; retained fitted directions have eigenvalue one.
    model_roots = linalg.eigvalsh(r, fitted.matrices["reproduced_correlation"])
    assert .5 * np.sum(np.square(model_roots - 1)) == pytest.approx(discrepancy, abs=3e-13)
    assert fitted.converged and fitted.diagnostics["local_minimum_certified"]
    assert fitted.diagnostics["gradient_max"] <= 1e-7
    assert fitted.diagnostics["minimum_objective_curvature"] > 0
    assert fitted.diagnostics["model_fit_test"] == "not provided"
    assert fitted.diagnostics["loading_inference"] == "not provided"
    assert len(fitted.diagnostics["optimizer_attempts"]) == 3


@pytest.mark.parametrize("seed", [410, 581])
def test_gls_analytic_gradient_matches_finite_differences(seed):
    r = correlation(seed)
    point = np.log(np.array([.2, .35, .25, .4, .29, .38]) - .005)
    _, native = gls_discrepancy(tensor(point), tensor(r), 2)

    def objective(theta):
        psi = .005 + np.exp(theta)
        scaled = r / np.sqrt(np.outer(psi, psi))
        tail = np.linalg.eigvalsh(scaled)[:-2]
        return np.sum((tail - 1) ** 2) / 2

    epsilon = 2e-5
    differences = [(objective(point + offset) - objective(point - offset)) / (2 * epsilon)
                   for offset in np.eye(len(point)) * epsilon]
    assert_allclose(native, differences, rtol=2e-8, atol=2e-10)


def test_gls_is_distinct_from_ml_and_minres_and_permutation_invariant():
    r = correlation()
    fitted = native_gls(r)
    smc = tensor(1 - 1 / np.linalg.inv(r).diagonal())
    ml = ex.maximum_likelihood(tensor(r), smc, 2, max_iter=1000)
    minres = ex.minimum_residual(tensor(r), smc, 2, max_iter=1000)
    assert np.max(np.abs(fitted.uniqueness.numpy() - ml.uniqueness.numpy())) > 1e-4
    assert np.max(np.abs(fitted.uniqueness.numpy() - minres.uniqueness.numpy())) > 1e-4
    # Optimized GLS uniqueness and one minus fitted communality are distinct.
    assert np.max(np.abs(fitted.uniqueness.numpy()
                         - (1 - fitted.loadings.square().sum(1).numpy()))) > 1e-5
    order = np.array([4, 0, 5, 2, 1, 3])
    permuted = native_gls(r[np.ix_(order, order)])
    assert_allclose(permuted.uniqueness, fitted.uniqueness[order], atol=2e-8)
    assert_allclose(permuted.loadings @ permuted.loadings.T,
                    (fitted.loadings @ fitted.loadings.T).numpy()[np.ix_(order, order)], atol=2e-8)


@pytest.mark.parametrize("seed", [410, 581])
@pytest.mark.parametrize("factors", [1, 2])
def test_generalized_image_against_generalized_eigenproblem(seed, factors):
    r = correlation(seed)
    inverse = np.linalg.inv(r)
    d = np.diag(1 / inverse.diagonal())
    roots, vectors = linalg.eigh(r, d)
    roots, vectors = roots[::-1], vectors[:, ::-1]
    # Generalized eigenvectors W satisfy W'DW=I. Production ordinary
    # eigenvectors V=S W, hence the independently expressed pattern is D W.
    reference = d @ vectors[:, :factors] @ np.diag((roots[:factors] - 1) / np.sqrt(roots[:factors]))
    fitted = generalized_image_factors(tensor(r), factors)
    assert_allclose(fitted.factored, roots, atol=2e-13)
    assert_allclose(fitted.loadings @ fitted.loadings.T, reference @ reference.T, atol=3e-13)
    assert_allclose(fitted.uniqueness, 1 - np.square(reference).sum(1), atol=3e-13)
    assert_allclose(fitted.matrices["conditional_variance"], d, atol=2e-14)
    assert_allclose(fitted.matrices["anti_image_covariance"], d @ inverse @ d, atol=2e-14)
    assert_allclose(fitted.matrices["image_covariance"], r - 2 * d + d @ inverse @ d, atol=3e-14)
    # The full generalized spectral reconstruction equals the image covariance,
    # even though the analysis retains only roots greater than one.
    all_loadings = d @ vectors @ np.diag((roots - 1) / np.sqrt(roots))
    assert_allclose(all_loadings @ all_loadings.T, fitted.matrices["image_covariance"], atol=3e-13)
    assert fitted.discrepancy is None and fitted.iterations == 0 and fitted.converged
    assert fitted.diagnostics["model_fit_test"] == "not provided"


def test_generalized_image_differs_from_image_covariance_pca_and_preserves_permutation():
    from openecon.econometrics.multivariate.extraction_extensions import image_covariance_factors
    r = correlation()
    fitted = generalized_image_factors(tensor(r), 2)
    old = image_covariance_factors(tensor(r), 2)
    assert float((fitted.loadings @ fitted.loadings.T - old.loadings @ old.loadings.T).abs().max()) > 1e-4
    order = np.array([4, 0, 5, 2, 1, 3])
    permuted = generalized_image_factors(tensor(r[np.ix_(order, order)]), 2)
    assert_allclose(permuted.loadings @ permuted.loadings.T,
                    (fitted.loadings @ fitted.loadings.T).numpy()[np.ix_(order, order)], atol=3e-13)


@pytest.mark.parametrize("bad", [torch.eye(2, dtype=torch.float64), torch.ones(3, 3, dtype=torch.float64),
                                tensor([[1, .5, .5], [.4, 1, .5], [.5, .5, 1]]),
                                tensor([[1, 2, 0], [2, 1, 0], [0, 0, 1]]),
                                tensor([[1, float("nan"), 0], [float("nan"), 1, 0], [0, 0, 1]]),
                                torch.eye(3), tensor([1, 1, 1]), [[1, 0, 0]]])
def test_additional_extractors_refuse_invalid_matrices(bad):
    with pytest.raises(KernelError):
        generalized_image_factors(bad, 1)
    with pytest.raises(KernelError):
        generalized_least_squares(bad, tensor([.5, .5, .5]), 1, max_iter=100)


@pytest.mark.parametrize("count", [None, True, 0, -1, 1.5, 6, 9])
def test_additional_extractors_require_explicit_identified_factor_count(count):
    with pytest.raises(KernelError):
        generalized_image_factors(tensor(correlation()), count)
    with pytest.raises(KernelError):
        native_gls(correlation(), factors=count)


@pytest.mark.parametrize("options", [{"max_iter": True}, {"max_iter": 0},
                                     {"max_iter": 10001}, {"tol": True},
                                     {"tol": float("nan")}, {"tol": 1e-6}])
def test_gls_rejects_invalid_options(options):
    with pytest.raises(KernelError) as error:
        native_gls(correlation(), **options)
    assert error.value.code == "invalid_option"


def test_gls_refuses_failed_iteration_and_work_geometry():
    with pytest.raises(KernelError) as error:
        native_gls(correlation(), max_iter=1)
    assert error.value.code == "no_convergence"
    with pytest.raises(KernelError) as error:
        native_gls(correlation(410, 33), factors=2)
    assert error.value.code == "workspace_limit"
    with pytest.raises(KernelError) as error:
        native_gls(correlation(410, 32), factors=2, max_iter=10000)
    assert error.value.code == "work_limit"


@pytest.mark.parametrize("r,count", [(np.eye(6), 1),
                                    (np.full((6, 6), .4) + np.eye(6) * .6, 2)])
def test_generalized_image_refuses_roots_at_one_or_unseparated(r, count):
    with pytest.raises(KernelError) as error:
        generalized_image_factors(tensor(r), count)
    assert error.value.code == "unidentified_factor"


def public_data():
    rng = np.random.default_rng(410)
    loadings = np.array([[.8, .12], [.7, .21], [.72, .08],
                         [.11, .82], [.18, .76], [.08, .7]])
    values = rng.normal(size=(231, 2)) @ loadings.T + rng.normal(size=(231, 6)) * .55
    return pd.DataFrame(values * np.arange(1, 7) + np.arange(10, 16),
                        columns=list("abcdef"),
                        index=pd.Index([f"subject-{i}" for i in range(231)], name="subject"))


@lru_cache(maxsize=None)
def public_fit(method, scores="regression"):
    return oe.factor(public_data(), list("abcdef"), method=method, factors=2, scores=scores)


def complete_tables_equal(one, two, atol=2e-7):
    assert set(one) == set(two)
    for key in one:
        assert one[key].index.equals(two[key].index), key
        assert one[key].columns.equals(two[key].columns), key
        assert_allclose(one[key], two[key], atol=atol, rtol=atol, equal_nan=True, err_msg=key)


@pytest.mark.parametrize("method", ["gls", "generalized_image"])
def test_public_extraction_complete_matrices_match_independent_oracle(method):
    fitted = public_fit(method)
    r = public_data().corr().to_numpy()
    loading = fitted["loadings"].to_numpy()
    uniqueness = fitted["uniqueness"].to_numpy()[:, 0]
    if method == "gls":
        psi, discrepancy = scipy_gls(r, 2)
        assert_allclose(uniqueness, psi, atol=2e-7, rtol=2e-6)
        reproduced = loading @ loading.T + np.diag(psi)
        assert_allclose(fitted["reproduced_correlation"], reproduced, atol=2e-7)
        assert_allclose(fitted["optimized_uniqueness"], np.diag(psi), atol=2e-7)
        assert fitted.attrs["discrepancy"] == pytest.approx(discrepancy, abs=3e-13)
        assert fitted.attrs["gradient_max"] <= 1e-7
        assert fitted.attrs["local_minimum_certified"]
    else:
        inverse = np.linalg.inv(r)
        d = np.diag(1 / inverse.diagonal())
        roots, vectors = linalg.eigh(r, d)
        reference = d @ vectors[:, -2:] @ np.diag((roots[-2:] - 1) / np.sqrt(roots[-2:]))
        assert_allclose(loading @ loading.T, reference @ reference.T, atol=3e-13)
        assert_allclose(fitted["conditional_variance"], d, atol=2e-14)
        assert_allclose(fitted["anti_image_covariance"], d @ inverse @ d, atol=3e-14)
        assert_allclose(fitted["image_covariance"], r - 2 * d + d @ inverse @ d, atol=3e-14)
    assert fitted.attrs["n"] == len(public_data())
    assert fitted.attrs["factors"] == 2 and fitted.attrs["mineigen"] is None
    assert fitted.attrs["model_fit_test"] == "not provided"
    assert fitted.attrs["loading_inference"] == "not provided"
    assert list(fitted["fit"].index) == ["independence"]
    assert fitted.attrs["device"] == "cpu" and fitted.attrs["precision"] == "float64"


@pytest.mark.parametrize("method", ["gls", "generalized_image"])
@pytest.mark.parametrize("scores", ["regression", "bartlett", "anderson_rubin"])
def test_public_score_coefficients_full_covariance_and_optimizer_free_replay(method, scores, monkeypatch):
    from openecon.econometrics.multivariate import extraction_additional
    from openecon.econometrics.summary_state import restore_summary, summary_state
    from openecon.engines import optimize as native_optimize

    fitted = public_fit(method, scores)
    data = public_data()
    r = data.corr().to_numpy()
    loading = fitted["loadings"].to_numpy()
    uniqueness = fitted["uniqueness"].to_numpy()[:, 0]
    weighted = loading / uniqueness[:, None]
    if scores == "regression":
        reference = np.linalg.solve(r, loading)
    elif scores == "bartlett":
        reference = weighted @ np.linalg.inv(loading.T @ weighted)
    else:
        values, vectors = linalg.eigh(weighted.T @ r @ weighted)
        reference = weighted @ vectors @ np.diag(1 / np.sqrt(values)) @ vectors.T
    assert_allclose(fitted["score_coefficients"], reference, atol=3e-13)
    expected_covariance = reference.T @ r @ reference
    if scores == "anderson_rubin":
        assert_allclose(expected_covariance, np.eye(2), atol=3e-13)
    query = data.iloc[[17, 2, 99, 0]].copy()
    query.iloc[1, 0] = np.nan
    expected = ((query - data.mean()) / data.std(ddof=1)).to_numpy() @ reference
    original_scores = oe.factor_scores(fitted, query)
    assert original_scores.index.equals(query.index)
    assert_allclose(original_scores, expected, atol=3e-13, equal_nan=True)
    state = summary_state(fitted)

    def forbid_refit(*args, **kwargs):
        raise AssertionError("Saved extraction/score replay must not refit")

    monkeypatch.setattr(extraction_additional, "generalized_least_squares", forbid_refit)
    monkeypatch.setattr(extraction_additional, "generalized_image_factors", forbid_refit)
    monkeypatch.setattr(native_optimize, "maximize_bfgs", forbid_refit)
    restored = restore_summary(state)
    assert summary_state(restored) == state
    assert restored.attrs == fitted.attrs
    complete_tables_equal(restored, fitted, atol=0)
    for key in fitted:
        assert restored[key].to_latex() == fitted[key].to_latex()
    assert_allclose(oe.factor_scores(restored, query), original_scores, atol=0, rtol=0, equal_nan=True)
    batches = oe.factor_scores(restored, oe.Dataset.from_frame(query)).iter_batches()
    assert_allclose(pd.concat(list(batches)), original_scores, atol=0, rtol=0, equal_nan=True)
    assert_allclose(restored["score_coefficients"].to_numpy().T @ r
                    @ restored["score_coefficients"].to_numpy(), expected_covariance, atol=3e-13)


@pytest.mark.parametrize("method", ["gls", "generalized_image"])
@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_public_summary_routes_preserve_all_tables_and_training_scores(method, matrix):
    data = public_data()
    values = data.cov() if matrix == "covariance" else data.corr()
    fitted = oe.factor_matrix(values, n=len(data), matrix=matrix, means=data.mean(),
                              sds=data.std(ddof=1), method=method, factors=2)
    expected = public_fit(method)
    complete_tables_equal(fitted, expected)
    assert fitted.attrs["input_kind"] == "summary_matrix"
    assert fitted.attrs["input_matrix"] == matrix
    assert fitted.attrs["n_missing"] == 0
    assert_allclose(oe.factor_scores(fitted, data), oe.factor_scores(expected, data), atol=2e-7)


@pytest.mark.parametrize("method", ["gls", "generalized_image"])
@pytest.mark.parametrize("frequency", [False, True])
def test_public_dataset_missing_and_frequency_alignment(method, frequency):
    data = public_data().assign(w=np.arange(231) % 4)
    data.iloc[2, 0] = np.nan
    data.iloc[4, -1] = np.nan
    names = list("abcdef")
    columns = names + ["w"] if frequency else names
    complete = data.dropna(subset=columns)
    expanded = (complete.loc[complete.index.repeat(complete.w.astype(int)), names]
                if frequency else complete[names])
    options = {"method": method, "factors": 2, "missing": "drop"}
    if frequency:
        options["weights"] = "w"
    streamed = oe.factor(oe.Dataset.from_frame(data), names, **options)
    expected = oe.factor(expanded, names, method=method, factors=2)
    complete_tables_equal(streamed, expected)
    assert streamed.attrs["n"] == len(expanded)
    assert streamed.attrs["n_missing"] == (2 if frequency else 1)
    assert streamed.attrs["source_content_sha256"]
    if frequency:
        assert streamed.attrs["weight_sum"] == int(complete.w.sum())
        assert streamed.attrs["physical_rows"] == int((complete.w > 0).sum())
        assert streamed.attrs["n_zero_weight"] == int((complete.w == 0).sum())
        assert streamed.attrs["covariance_divisor"] == len(expanded) - 1
    assert_allclose(oe.factor_scores(streamed, data), oe.factor_scores(expected, data),
                    atol=2e-7, equal_nan=True)
    with pytest.raises(AnalysisError):
        oe.factor(oe.Dataset.from_frame(data), names, method=method, factors=2,
                   missing="raise", **({"weights": "w"} if frequency else {}))


@pytest.mark.parametrize("method", ["gls", "generalized_image"])
@pytest.mark.parametrize("options", [{}, {"mineigen": .01, "factors": 2}, {"factors": 6}])
def test_public_extractors_refuse_automatic_or_unidentified_counts(method, options):
    with pytest.raises(AnalysisError):
        oe.factor(public_data(), list("abcdef"), method=method, **options)


@pytest.mark.parametrize("options", [{"tolerance": 1e-7}, {"max_iterations": 1000}])
def test_public_generalized_image_refuses_ignored_closed_form_controls(options):
    with pytest.raises(AnalysisError, match="closed form"):
        oe.factor(public_data(), list("abcdef"), method="generalized_image", factors=2, **options)


@pytest.mark.parametrize("variables,options,code", [(33, {}, "workspace_limit"),
                                                   (32, {"max_iterations": 10000}, "work_limit"),
                                                   (6, {"tolerance": 1e-6}, "invalid_option"),
                                                   (6, {"tolerance": "bad"}, "invalid_option"),
                                                   (6, {"tolerance": 1e-8 + 0j}, "invalid_option"),
                                                   (6, {"tolerance": True}, "invalid_option")])
def test_public_gls_admission_precedes_dataset_read(variables, options, code):
    names = [f"v{i}" for i in range(variables)]

    def unreadable_batches():
        raise AssertionError("Invalid GLS geometry must be rejected before reading data")
        yield pd.DataFrame()

    data = oe.Dataset.from_batches(unreadable_batches, columns=names, row_count=231)
    with pytest.raises(AnalysisError) as error:
        oe.factor(data, names, method="gls", factors=2, **options)
    assert error.value.code == code
