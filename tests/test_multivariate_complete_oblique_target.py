"""Complete oblique targets: exact recovery and an independent angle oracle."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import least_squares

import openecon as oe
from openecon.econometrics.multivariate import rotation_extensions as ext
from openecon.engines.contracts import KernelError


def loadings(factors=2):
    values = [[-.81, .11, .15], [-.70, -.07, -.13], [-.74, .18, .18],
              [-.08, .62, .10], [.05, .58, .05], [-.17, .67, -.09],
              [.10, .12, .80], [-.14, -.13, .70], [.03, -.08, .90]]
    return torch.tensor(values, dtype=torch.float64)[:, :factors]


def oblique_matrix(first, second):
    # Independent two-angle parametrization of the unit-column T manifold.
    transform = np.array([[np.cos(first), -np.sin(second)],
                          [np.sin(first), np.cos(second)]])
    return np.linalg.inv(transform).T


def assert_invariants(original, result):
    rotated = result.rotation
    np.testing.assert_allclose(rotated.pattern, original @ rotated.matrix, atol=1e-12)
    np.testing.assert_allclose(rotated.phi.diagonal(), 1., atol=1e-10)
    np.testing.assert_allclose(rotated.pattern @ rotated.phi @ rotated.pattern.T,
                               original @ original.T, atol=2e-10, rtol=2e-10)
    assert rotated.oblique
    assert result.diagnostics["target_complete"] is True
    assert result.diagnostics["target_specified"] == original.numel()
    assert result.diagnostics["target_jacobian_rank"] == original.shape[1] * (original.shape[1] - 1)
    assert result.diagnostics["rotation_projected_gradient"] <= 1e-5
    assert result.diagnostics["rotation_start"] == "identity"
    assert "minimum/global uniqueness not established" in result.diagnostics["rotation_optimum"]
    assert bool(result.mask.all())
    assert bool(torch.isfinite(result.target).all())


@pytest.mark.parametrize("factors", [2, 3])
@pytest.mark.parametrize("kaiser", [False, True])
def test_full_target_exact_constructed_common_covariance(factors, kaiser):
    original = loadings(factors)
    transform = torch.tensor([[1., .13, -.07], [.18, 1., .11], [-.12, .08, 1.]],
                             dtype=torch.float64)[:factors, :factors]
    transform = transform / transform.square().sum(0).sqrt()
    expected_matrix = torch.linalg.inv(transform).T
    target = original @ expected_matrix
    result = ext.oblique_target(original, target=target, kaiser=kaiser)
    np.testing.assert_allclose(result.rotation.pattern, target, atol=3e-8)
    np.testing.assert_allclose(result.rotation.matrix, expected_matrix, atol=3e-8)
    # Axis order and negative dominant first-column orientation remain the caller's.
    assert float(result.rotation.pattern[result.rotation.pattern[:, 0].abs().argmax(), 0]) < 0
    np.testing.assert_array_equal(result.target, target)
    assert result.diagnostics["rotation_criterion"] < 1e-14
    assert_invariants(original, result)


@pytest.mark.parametrize("kaiser", [False, True])
def test_complete_noisy_target_matches_independent_scipy_constrained_oracle(kaiser):
    original = loadings()
    target = original.numpy() @ oblique_matrix(.18, -.23)
    target += np.array([[.02, -.03], [-.01, .02], [.04, .01], [-.03, -.02],
                        [.01, -.01], [.02, .04], [-.01, .01], [.03, -.02], [-.02, .01]])
    actual = ext.oblique_target(original, target=target, kaiser=kaiser)
    scale = np.sqrt(np.sum(original.numpy() ** 2, axis=1)) if kaiser else np.ones(len(original))
    a, h = original.numpy() / scale[:, None], target / scale[:, None]

    def residual(angles):
        return (a @ oblique_matrix(*angles) - h).ravel()

    reference = least_squares(residual, [0., 0.], jac="3-point", ftol=1e-13,
                              xtol=1e-13, gtol=1e-13, max_nfev=1000)
    assert reference.success and reference.optimality < 1e-8
    np.testing.assert_allclose(actual.rotation.pattern,
                               original.numpy() @ oblique_matrix(*reference.x), atol=3e-8)
    assert actual.diagnostics["rotation_criterion"] == pytest.approx(np.dot(reference.fun, reference.fun) / 2,
                                                                       abs=1e-12)
    assert_invariants(original, actual)


@pytest.mark.parametrize("bad", ["nan", "inf", "complex", "rank", "near_rank", "zero", "shape"])
def test_complete_target_refuses_missing_or_unidentified_cells(bad):
    original = loadings()
    target = original.numpy().copy()
    if bad == "nan":
        target[0, 1] = np.nan
    elif bad == "inf":
        target[0, 1] = np.inf
    elif bad == "complex":
        target = target.astype(complex) + 1j
    elif bad == "rank":
        target[:, 1] = target[:, 0]
    elif bad == "near_rank":
        target[:, 1] = target[:, 0] + 1e-12 * target[:, 1]
    elif bad == "zero":
        target[:, 1] = 0.
    else:
        target = target[:-1]
    with pytest.raises(KernelError):
        ext.oblique_target(original, target=target)


def test_complete_target_preserves_failure_to_converge():
    original = loadings()
    target = original.numpy() @ oblique_matrix(.61, -.43)
    with pytest.raises(KernelError, match="did not converge"):
        ext.oblique_target(original, target=target, max_iter=1)


@pytest.mark.parametrize("option,value", [("tol", 0.), ("tol", True), ("kaiser", 1), ("max_iter", 0)])
def test_complete_target_options_are_validated(option, value):
    original = loadings()
    with pytest.raises(KernelError):
        ext.oblique_target(original, target=original, **{option: value})


@pytest.mark.parametrize("kaiser", [False, True])
def test_public_complete_target_keeps_local_diagnostics_and_saved_scores(kaiser):
    rng = np.random.default_rng(1011)
    latent = rng.normal(size=(350, 2))
    a = loadings().numpy()
    frame = pd.DataFrame(latent @ a.T + rng.normal(size=(350, len(a))) * .65,
                         columns=[f"v{j}" for j in range(len(a))])
    names = list(frame)
    base = oe.factor(frame, names, factors=2)
    target = base["loadings"].to_numpy() @ oblique_matrix(.18, -.23)
    result = oe.factor(frame, names, factors=2, rotate="target", target=target,
                       target_oblique=True, kaiser=kaiser, rotation_tolerance=1e-9)
    assert result.attrs["target_complete"] is True
    assert result.attrs["rotation_optimum"] == "local stationary point; minimum/global uniqueness not established"
    assert result.attrs["rotation_tolerance"] == 1e-9
    np.testing.assert_allclose(result["rotated_loadings"], target, atol=3e-8)
    phi = result["factor_correlations"].to_numpy()
    np.testing.assert_allclose(result["structure"], target @ phi, atol=3e-8)
    np.testing.assert_allclose(result["score_coefficients"],
                               np.linalg.solve(np.corrcoef(frame.to_numpy(), rowvar=False), target @ phi), atol=3e-8)
    restored = oe.restore_summary(oe.summary_state(result))
    query = frame.iloc[[7, 2, 7, 19]].copy()
    query.iloc[1, 0] = np.nan
    pd.testing.assert_frame_equal(oe.factor_scores(restored, query), oe.factor_scores(result, query))
    for table in result:
        pd.testing.assert_frame_equal(restored[table], result[table])
