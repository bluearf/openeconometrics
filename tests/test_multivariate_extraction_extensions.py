"""Independent NumPy and linear-regression oracles for extraction kernels.

Published alpha fixture: SAS/IML ALPHA sample library, Harman's eight physical
variables; https://support.sas.com/documentation/onlinedoc/iml/ex_code/143/alpha.html
The separate published output is in SAS/IML User's Guide, Example 10.4.
https://support.sas.com/documentation/cdl/en/imlug/68150/HTML/default/imlug_genstatexpls_sect006.htm
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from openecon.engines.contracts import KernelError
from openecon.econometrics.multivariate.extraction_extensions import (
    alpha_factors, image_covariance_factors,
)


HARMAN = np.array([
    [1, .846, .805, .859, .473, .398, .301, .382],
    [.846, 1, .881, .826, .376, .326, .277, .415],
    [.805, .881, 1, .801, .380, .319, .237, .345],
    [.859, .826, .801, 1, .436, .329, .327, .365],
    [.473, .376, .380, .436, 1, .762, .730, .629],
    [.398, .326, .319, .329, .762, 1, .583, .577],
    [.301, .277, .237, .327, .730, .583, 1, .539],
    [.382, .415, .345, .365, .629, .577, .539, 1],
])
PUBLISHED_COMMUNALITIES = np.array([
    .8381205, .8905717, .81893, .8067292, .8802149, .6391977, .5821583, .4998126,
])
PUBLISHED_LOADINGS = np.array([
    [.813386, -.420147], [.8028363, -.49601], [.7579087, -.494474],
    [.7874461, -.432039], [.8051439, .4816205], [.6804127, .4198051],
    [.620623, .4438303], [.6449419, .2895902],
])


def matrix(value):
    return torch.tensor(value, dtype=torch.float64)


def smc(r):
    return 1 - 1 / np.linalg.inv(r).diagonal()


def observations(seed):
    rng = np.random.default_rng(seed)
    # Different correlation structures and unequal original measurement scales.
    loadings = np.array([[.82, .06], [.76, .18], [.68, .11],
                         [.12, .77], [.18, .69], [.05, .84]])
    x = rng.normal(size=(241, 2)) @ loadings.T + rng.normal(size=(241, 6)) * .5
    x = x * np.arange(1, 7) + np.arange(30, 36)
    return (x - x.mean(0)) / x.std(0, ddof=1)


def numpy_alpha(r, m, tolerance):
    h = smc(r)
    for iteration in range(1, 1001):
        g = (r - np.eye(len(r))) / np.sqrt(np.outer(h, h)) + np.eye(len(r))
        values, vectors = np.linalg.eigh(g)
        values, vectors = values[::-1], vectors[:, ::-1]
        loadings = np.sqrt(h)[:, None] * vectors[:, :m] * np.sqrt(values[:m])
        updated = np.square(loadings).sum(1)
        change = np.max(np.abs(updated - h))
        if change <= tolerance:
            return values, loadings, updated, iteration, change
        h = updated
    raise AssertionError("Oracle failed to converge")


@pytest.mark.parametrize("seed", [29, 341])
def test_alpha_independent_iteration_and_reconstructed_covariance(seed):
    r = np.corrcoef(observations(seed), rowvar=False)
    roots, reference, h, iterations, change = numpy_alpha(r, 2, 1e-8)
    fit = alpha_factors(matrix(r), matrix(smc(r)), 2, max_iter=1000, tol=1e-8)
    np.testing.assert_allclose(fit.factored, roots, atol=2e-10)
    np.testing.assert_allclose(fit.loadings @ fit.loadings.T, reference @ reference.T, atol=2e-10)
    np.testing.assert_allclose(fit.uniqueness, 1 - h, atol=2e-10)
    assert fit.iterations == iterations
    assert fit.diagnostics["communality_max_change"] == pytest.approx(change, abs=2e-14)
    assert fit.diagnostics["converged"] and fit.converged
    assert fit.discrepancy is None
    assert fit.diagnostics["model_fit_test"] == "not provided"


def test_alpha_published_sas_harman_fixture():
    fit = alpha_factors(matrix(HARMAN), matrix(smc(HARMAN)), 2, max_iter=1000, tol=.001)
    np.testing.assert_allclose(1 - fit.uniqueness, PUBLISHED_COMMUNALITIES, atol=6e-7, rtol=0)
    # Whole-column signs are unidentified; compare the full published common covariance.
    np.testing.assert_allclose(fit.loadings @ fit.loadings.T,
                               PUBLISHED_LOADINGS @ PUBLISHED_LOADINGS.T, atol=1.4e-6, rtol=0)
    np.testing.assert_allclose(fit.factored[:2], [5.937855, 2.0621956], atol=6e-7, rtol=0)
    assert fit.diagnostics["retained_roots_above_one"]


@pytest.mark.parametrize("seed", [29, 341])
@pytest.mark.parametrize("count", [1, 2, 6])
def test_image_covariance_independent_leave_one_variable_out_regressions(seed, count):
    z = observations(seed)
    r = np.corrcoef(z, rowvar=False)
    coefficients = np.zeros_like(r)
    for target in range(z.shape[1]):
        other = np.arange(z.shape[1]) != target
        coefficients[other, target] = np.linalg.lstsq(z[:, other], z[:, target], rcond=None)[0]
    predictions = z @ coefficients
    covariance = np.cov(predictions, rowvar=False)
    roots, axes = np.linalg.eigh(covariance)
    roots, axes = roots[::-1], axes[:, ::-1]
    reference = axes[:, :count] * np.sqrt(roots[:count])
    fit = image_covariance_factors(matrix(r), count)
    np.testing.assert_allclose(fit.matrices["image_coefficients"], coefficients, atol=2e-13)
    np.testing.assert_allclose(fit.matrices["image_covariance"], covariance, atol=2e-13)
    np.testing.assert_allclose(fit.factored, roots, atol=2e-13)
    np.testing.assert_allclose(fit.loadings @ fit.loadings.T, reference @ reference.T, atol=3e-13)
    np.testing.assert_allclose(fit.uniqueness, 1 - np.square(reference).sum(1), atol=3e-13)
    assert fit.iterations == 0 and fit.discrepancy is None
    assert fit.diagnostics["extraction_variant"] == "SAS/Guttman image covariance PCA"
    assert fit.diagnostics["model_fit_test"] == "not provided"


def test_image_inverse_identity_and_permutation():
    r = HARMAN
    fit = image_covariance_factors(matrix(r), 2)
    inverse = np.linalg.inv(r)
    d = np.diag(1 / inverse.diagonal())
    np.testing.assert_allclose(fit.matrices["image_covariance"], r - 2 * d + d @ inverse @ d, atol=3e-14)
    order = np.array([5, 0, 7, 3, 1, 6, 2, 4])
    permuted = image_covariance_factors(matrix(r[np.ix_(order, order)]), 2)
    np.testing.assert_allclose(permuted.factored, fit.factored, atol=2e-14)
    np.testing.assert_allclose(permuted.loadings @ permuted.loadings.T,
                               (fit.loadings @ fit.loadings.T).numpy()[np.ix_(order, order)], atol=3e-14)


@pytest.mark.parametrize("bad", [torch.eye(2, dtype=torch.float64), torch.ones(3, 3, dtype=torch.float64),
                                matrix([[1, .5, .5], [.4, 1, .5], [.5, .5, 1]]),
                                matrix([[2, .5, .5], [.5, 1, .5], [.5, .5, 1]]),
                                matrix([[1, 2, 0], [2, 1, 0], [0, 0, 1]]),
                                matrix([[1, float("nan"), 0], [float("nan"), 1, 0], [0, 0, 1]]),
                                torch.eye(3), matrix([1, 1, 1]), [[1, 0, 0]]])
def test_both_kernels_reject_invalid_correlation(bad):
    with pytest.raises(KernelError):
        alpha_factors(bad, matrix([.5, .5, .5]), 1, max_iter=100, tol=.001)
    with pytest.raises(KernelError):
        image_covariance_factors(bad, 1)


@pytest.mark.parametrize("count", [None, True, 0, -1, 1.5, 9])
def test_both_require_identified_explicit_factor_counts(count):
    with pytest.raises(KernelError):
        alpha_factors(matrix(HARMAN), matrix(smc(HARMAN)), count, max_iter=100, tol=.001)
    with pytest.raises(KernelError):
        image_covariance_factors(matrix(HARMAN), count)


@pytest.mark.parametrize("initial", [np.zeros(8), np.full(8, -1), np.full(8, 1.1),
                                    np.full(8, np.inf), np.ones(7)])
def test_alpha_initial_communalities_are_guarded(initial):
    with pytest.raises(KernelError):
        alpha_factors(matrix(HARMAN), matrix(initial), 2, max_iter=100, tol=.001)


@pytest.mark.parametrize("options", [{"max_iter": 0, "tol": .001}, {"max_iter": True, "tol": .001},
                                    {"max_iter": 100, "tol": 0}, {"max_iter": 100, "tol": np.inf},
                                    {"max_iter": 100, "tol": True}])
def test_alpha_iteration_options_are_guarded(options):
    with pytest.raises(KernelError):
        alpha_factors(matrix(HARMAN), matrix(smc(HARMAN)), 2, **options)


def test_alpha_iteration_limit_unidentified_roots_and_no_mutation():
    r, h = matrix(HARMAN), matrix(smc(HARMAN))
    before_r, before_h = r.clone(), h.clone()
    with pytest.raises(KernelError, match="did not converge"):
        alpha_factors(r, h, 2, max_iter=1, tol=1e-14)
    with pytest.raises(KernelError, match="roots"):
        alpha_factors(r, h, 8, max_iter=100, tol=.001)
    fit = alpha_factors(r, h, 2, max_iter=100, tol=.001)
    image_covariance_factors(r, 2)
    assert fit.heywood == []
    torch.testing.assert_close(r, before_r, rtol=0, atol=0)
    torch.testing.assert_close(h, before_h, rtol=0, atol=0)


def test_alpha_reports_unclipped_heywood_and_last_iteration_spectrum():
    r = np.array([[1, .9097383549291681, .6631107226039137, .07129214441347896],
                  [.9097383549291681, 1, .5194631546593375, .20557632582257435],
                  [.6631107226039137, .5194631546593375, 1, .09434708339814853],
                  [.07129214441347896, .20557632582257435, .09434708339814853, 1]])
    fit = alpha_factors(matrix(r), matrix(smc(r)), 1, max_iter=1000, tol=1e-5)
    assert fit.heywood == [1]
    assert fit.uniqueness[1] < -.3
    np.testing.assert_allclose(fit.uniqueness, 1 - fit.loadings.square().sum(1), atol=0, rtol=0)
    last = np.array(fit.diagnostics["last_iteration_communalities"])
    g = (r - np.eye(len(r))) / np.sqrt(np.outer(last, last)) + np.eye(len(r))
    np.testing.assert_allclose(fit.factored, np.linalg.eigvalsh(g)[::-1], atol=1e-13)
    assert fit.diagnostics["communality_max_change"] <= 1e-5


def test_null_image_and_matrix_workspaces_are_explicit_gates():
    with pytest.raises(KernelError, match="image variance"):
        image_covariance_factors(torch.eye(3, dtype=torch.float64), 1)
    with pytest.raises(KernelError, match="positive finite initial"):
        alpha_factors(torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
                      1, max_iter=10, tol=.001)
    with pytest.raises(KernelError, match="work budget"):
        alpha_factors(matrix(HARMAN), matrix(smc(HARMAN)), 2, max_iter=4_000_000, tol=.001)
    with pytest.raises(KernelError, match="256 variables"):
        image_covariance_factors(torch.eye(257, dtype=torch.float64), 1)
