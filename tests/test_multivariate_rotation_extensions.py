"""Independent rotation losses, domain fixtures, tangent checks and failure gates."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.optimize import minimize, minimize_scalar

from openecon.econometrics.multivariate import rotation
from openecon.econometrics.multivariate import rotation_extensions as ext
from openecon.engines.contracts import KernelError


def domain(name):
    # Process quality and respondent constructs have distinct, signed cross-loadings.
    if name == "process_quality":
        values = [[.81, .11], [.70, -.07], [.74, .18], [.08, .82], [-.05, .68], [.17, .77]]
    else:
        values = [[.70, .24], [.83, .11], [.63, -.12], [.35, .71], [.11, .80], [-.14, .65]]
    return torch.tensor(values, dtype=torch.float64) @ orth(.39)


def orth(angle):
    return torch.tensor([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]], dtype=torch.float64)


def independent_cf(loadings, kappa):
    # Explicit pair sums, independent of the production broadcast implementation.
    p, m = loadings.shape
    row = sum(loadings[i, j]**2 * loadings[i, k]**2 for i in range(p)
              for j in range(m) for k in range(m) if j != k)
    column = sum(loadings[i, j]**2 * loadings[k, j]**2 for j in range(m)
                 for i in range(p) for k in range(p) if i != k)
    return ((1 - kappa) * row + kappa * column) / 4


def independent_partial(loadings, target, mask):
    return sum((loadings[i, j] - target[i, j])**2 / 2 for i, j in zip(*np.where(mask), strict=True))


def angle_matrix(angles):
    first, second = angles
    t = np.array([[np.cos(first), -np.sin(second)], [np.sin(first), np.cos(second)]])
    if abs(np.linalg.det(t)) < 1e-6:
        return None
    return np.linalg.inv(t).T


def canonical(pattern):
    order = np.argsort(-np.sum(pattern**2, axis=0), kind="stable")
    pattern = pattern[:, order]
    return pattern * np.sign(pattern[np.argmax(abs(pattern), axis=0), np.arange(pattern.shape[1])])


def invariants(loadings, result):
    rotated = result.rotation
    np.testing.assert_allclose(rotated.pattern, loadings @ rotated.matrix, atol=2e-12)
    np.testing.assert_allclose(rotated.pattern @ rotated.phi @ rotated.pattern.T,
                               loadings @ loadings.T, atol=2e-10)
    np.testing.assert_allclose(rotated.phi.diagonal(), 1., atol=2e-10)
    if not rotated.oblique:
        np.testing.assert_allclose(rotated.matrix.T @ rotated.matrix, np.eye(loadings.shape[1]), atol=2e-12)
    assert result.diagnostics["rotation_projected_gradient"] <= 1e-5
    assert result.diagnostics["rotation_start"] == "identity"
    assert "minimum/global uniqueness not established" in result.diagnostics["rotation_optimum"]


@pytest.mark.parametrize("kappa", [0., .17, .63, 1.])
def test_cf_gradient_independent_pair_loss(kappa):
    rng = np.random.default_rng(733)
    values = rng.normal(size=(7, 3))
    actual, gradient = ext.cf_value(torch.tensor(values), kappa)
    assert actual == pytest.approx(independent_cf(values, kappa), rel=2e-14)
    delta = 1e-6
    numeric = np.empty_like(values)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += delta
        minus[index] -= delta
        numeric[index] = (independent_cf(plus, kappa) - independent_cf(minus, kappa)) / (2 * delta)
    np.testing.assert_allclose(gradient, numeric, atol=2e-8, rtol=2e-8)


def test_partial_gradient_and_free_cells():
    rng = np.random.default_rng(825)
    values, target = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    mask = rng.random((8, 3)) > .4
    value, gradient = ext.partial_value(torch.tensor(values), torch.tensor(target), torch.tensor(mask))
    assert value == pytest.approx(independent_partial(values, target, mask))
    numeric = np.empty_like(values)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numeric[index] = (independent_partial(plus, target, mask) - independent_partial(minus, target, mask)) / 2e-6
    np.testing.assert_allclose(gradient, numeric, atol=2e-9)
    assert bool((gradient[~torch.tensor(mask)] == 0).all())


@pytest.mark.parametrize("name", ["process_quality", "respondent_constructs"])
@pytest.mark.parametrize("kaiser", [False, True])
@pytest.mark.parametrize("kappa", [0., 1 / 6, .37, 1.])
def test_orthogonal_cf_independent_angle_oracle(name, kaiser, kappa):
    loadings = domain(name)
    result = ext.cf(loadings, kappa=kappa, kaiser=kaiser)
    a = loadings.numpy()
    if kaiser:
        a = a / np.sqrt(np.sum(a**2, axis=1))[:, None]
    def loss(angle):
        return independent_cf(a @ orth(angle).numpy(), kappa)
    grid = np.linspace(-np.pi, np.pi, 401)
    best = grid[np.argmin([loss(angle) for angle in grid])]
    oracle = minimize_scalar(loss, bounds=(best - .02, best + .02), method="bounded",
                             options={"xatol": 1e-13})
    assert result.diagnostics["rotation_criterion"] == pytest.approx(oracle.fun, abs=2e-9)
    np.testing.assert_allclose(result.rotation.pattern,
                               canonical(loadings.numpy() @ orth(oracle.x).numpy()), atol=2e-7)
    invariants(loadings, result)


@pytest.mark.parametrize("name", ["process_quality", "respondent_constructs"])
@pytest.mark.parametrize("kappa,method", [(0., "quartimax"), (1 / 6, "varimax"), (1 / 6, "equamax")])
def test_cf_orthomax_special_cases(name, kappa, method):
    loadings = domain(name)
    result = ext.cf(loadings, kappa=kappa)
    reference = rotation.rotate(loadings, method)
    np.testing.assert_allclose(result.rotation.pattern, reference.pattern, atol=2e-7)


@pytest.mark.parametrize("name", ["process_quality", "respondent_constructs"])
@pytest.mark.parametrize("kaiser", [False, True])
@pytest.mark.parametrize("kappa", [0., .37, 1.])
def test_oblique_cf_independent_unit_column_oracle(name, kaiser, kappa):
    loadings = domain(name)
    result = ext.cf(loadings, kappa=kappa, oblique=True, kaiser=kaiser)
    a = loadings.numpy()
    if kaiser:
        a = a / np.sqrt(np.sum(a**2, axis=1))[:, None]
    def loss(angles):
        matrix = angle_matrix(angles)
        return 1e100 if matrix is None else independent_cf(a @ matrix, kappa)
    oracles = [minimize(loss, start, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
               for start in ([0., 0.], [.15, -.25], [-.25, .15])]
    best = min(oracles, key=lambda fit: fit.fun)
    assert result.diagnostics["rotation_criterion"] == pytest.approx(best.fun, abs=3e-8)
    np.testing.assert_allclose(result.rotation.pattern,
                               canonical(loadings.numpy() @ angle_matrix(best.x)), atol=3e-6)
    invariants(loadings, result)
    if kappa == 0:
        reference = rotation.rotate(loadings, "oblimin", normalize=kaiser, max_iter=10000)
        np.testing.assert_allclose(result.rotation.pattern, reference.pattern, atol=2e-7)


@pytest.mark.parametrize("name", ["process_quality", "respondent_constructs"])
@pytest.mark.parametrize("oblique", [False, True])
@pytest.mark.parametrize("kaiser", [False, True])
def test_partial_target_planted_rotation_and_independent_oracle(name, oblique, kaiser):
    loadings = domain(name)
    planted = torch.tensor(angle_matrix([.18, -.23])) if oblique else orth(-.27)
    target = loadings @ planted
    mask = torch.tensor([[1, 0], [1, 1], [0, 1], [1, 0], [1, 1], [0, 1]], dtype=torch.bool)
    result = ext.partial_target(loadings, target=target, mask=mask, oblique=oblique, kaiser=kaiser)
    np.testing.assert_allclose(result.rotation.pattern, target, atol=2e-8)
    np.testing.assert_allclose(result.rotation.matrix, planted, atol=2e-8)
    a, h = loadings.numpy(), target.numpy()
    if kaiser:
        scale = np.sqrt(np.sum(a**2, axis=1))[:, None]
        a, h = a / scale, h / scale
    def loss(angles):
        matrix = angle_matrix(angles) if oblique else orth(float(angles[0])).numpy()
        return 1e100 if matrix is None else independent_partial(a @ matrix, h, mask.numpy())
    oracle = minimize(loss, [0., 0.] if oblique else [0.], method="BFGS",
                       options={"gtol": 1e-9, "maxiter": 1000})
    assert result.diagnostics["rotation_criterion"] == pytest.approx(oracle.fun, abs=1e-12)
    assert result.diagnostics["target_jacobian_rank"] == (2 if oblique else 1)
    assert result.diagnostics["target_specified"] == 8
    assert torch.isnan(result.target[~mask]).all()
    ignored = target.clone()
    ignored[~mask] = float("inf")
    repeated = ext.partial_target(loadings, target=ignored, mask=mask, oblique=oblique, kaiser=kaiser)
    np.testing.assert_array_equal(result.rotation.pattern, repeated.rotation.pattern)
    invariants(loadings, result)


@pytest.mark.parametrize("oblique", [False, True])
def test_tangent_jacobian_central_difference(oblique):
    a = domain("respondent_constructs")
    matrix = torch.tensor(angle_matrix([.18, -.23])) if oblique else orth(-.27)
    pattern = a @ matrix
    mask = torch.tensor([[1, 0], [1, 1], [0, 1], [1, 0], [1, 1], [0, 1]], dtype=torch.bool)
    actual = ext.tangent_jacobian(pattern, matrix, mask, oblique=oblique)
    numeric = []
    delta = 1e-6
    if not oblique:
        plus, minus = a @ matrix @ orth(delta / math.sqrt(2)), a @ matrix @ orth(-delta / math.sqrt(2))
        # orth() has the opposite elementary skew sign from the production basis.
        numeric.append(((minus - plus) / (2 * delta))[mask])
    else:
        t = torch.linalg.inv(matrix).T
        for column in range(2):
            _, _, vh = torch.linalg.svd(t[:, column][None, :], full_matrices=True)
            direction = torch.zeros_like(t)
            direction[:, column] = vh[1]
            plus, minus = t + delta * direction, t - delta * direction
            plus, minus = plus / plus.square().sum(0).sqrt(), minus / minus.square().sum(0).sqrt()
            numeric.append(((a @ torch.linalg.inv(plus).T - a @ torch.linalg.inv(minus).T) / (2 * delta))[mask])
    np.testing.assert_allclose(actual, torch.stack(numeric, dim=1), atol=2e-10)


@pytest.mark.parametrize("bad", [-.1, 1.1, float("nan"), float("inf"), True])
def test_cf_parameter_gates(bad):
    with pytest.raises(KernelError):
        ext.cf(domain("process_quality"), kappa=bad)


@pytest.mark.parametrize("bad", [0, 10001, True, 1.5])
def test_iteration_gates(bad):
    with pytest.raises(KernelError):
        ext.cf(domain("process_quality"), max_iter=bad)


@pytest.mark.parametrize("bad", [0., 1e-14, 1e-4, float("nan"), True])
def test_tolerance_gates(bad):
    with pytest.raises(KernelError):
        ext.cf(domain("process_quality"), tol=bad)


def test_loading_device_precision_and_condition_gates():
    loadings = domain("process_quality")
    for bad in (loadings.float(), loadings.to(torch.complex128), loadings[:1], loadings * float("inf"),
                torch.ones((6, 2), dtype=torch.float64), torch.eye(17, dtype=torch.float64),
                torch.ones((257, 2), dtype=torch.float64)):
        with pytest.raises(KernelError):
            ext.cf(bad)
    if torch.backends.mps.is_available():
        with pytest.raises(KernelError):
            ext.cf(loadings.float().to("mps"))
    with pytest.raises(KernelError, match="did not converge"):
        ext.cf(loadings, max_iter=1)


@pytest.mark.parametrize("mask", [np.zeros((6, 2)), np.ones((5, 2)), np.full((6, 2), .5),
                                  np.full((6, 2), np.nan), np.full((6, 2), np.inf)])
def test_partial_mask_gates(mask):
    with pytest.raises(KernelError):
        ext.partial_target(domain("process_quality"), target=np.ones((6, 2)), mask=mask)


def test_partial_target_anchor_shape_and_rank_gates():
    loadings = domain("process_quality")
    mask = torch.ones_like(loadings, dtype=torch.bool)
    for target in (loadings[:1], loadings * float("nan"), torch.zeros_like(loadings),
                   loadings.to(torch.complex128)):
        with pytest.raises(KernelError):
            ext.partial_target(loadings, target=target, mask=mask)
    # Three factors but only one targeted row: three constraints cannot identify
    # the three orthogonal tangent dimensions (the row supplies only two).
    dense = torch.tensor([[.6, .3, .2], [.2, .6, .1], [.1, .2, .7],
                          [.7, -.2, .1], [.1, .7, -.2], [-.2, .1, .6]], dtype=torch.float64)
    mask = torch.zeros_like(dense, dtype=torch.bool)
    mask[0] = True
    with pytest.raises(KernelError, match="locally identify"):
        ext.partial_target(dense, target=dense, mask=mask, kaiser=False)
    with pytest.raises(KernelError, match="fewer"):
        ext.partial_target(dense, target=dense, mask=mask, oblique=True, kaiser=False)
    near = loadings.clone()
    near[:, 1] = near[:, 0] + 1e-12 * near[:, 1]
    with pytest.raises(KernelError, match="ill-conditioned"):
        ext.partial_target(near, target=loadings, mask=torch.ones_like(loadings, dtype=torch.bool))


def test_oblique_near_singular_transform_refuses_without_phi_repair():
    loadings = domain("process_quality")
    matrix = torch.tensor([[1., 1.], [0., 1e-12]], dtype=torch.float64)
    with pytest.raises(KernelError, match="ill-conditioned"):
        ext._result(loadings, matrix, 0, True, oblique=True, canonical=False)
