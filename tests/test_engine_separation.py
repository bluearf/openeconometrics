"""Independent LP certificates and adversarial native separation contracts."""
from __future__ import annotations

import builtins
import itertools

import numpy as np
import pytest
from scipy.optimize import linprog
import torch

from openecon.analysis import _check_binary_separation
from openecon.analysis_contracts import AnalysisError
from openecon.engines import separation
from openecon.engines.contracts import KernelError


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


def replay(rows, batch=7):
    def source():
        for start in range(0, len(rows), batch):
            yield rows[start:start + batch]
    return source


def expected_lp(rows):
    a = rows.numpy()
    result = linprog(-a.mean(0), A_ub=-a, b_ub=np.zeros(len(a)),
                     bounds=[(-1, 1)] * a.shape[1], method="highs",
                     options={"primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    assert result.success
    return -result.fun


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8, 12])
@pytest.mark.parametrize("kind", ["overlap", "complete", "quasi", "collinear"])
@pytest.mark.parametrize("seed", [41, 93, 129])
def test_classification_against_independent_highs_oracle(width, kind, seed):
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn((60, width), generator=generator, dtype=torch.float64)
    if kind == "complete":
        rows[:, 0] = rows[:, 0].abs() + .2
    elif kind == "quasi":
        rows[:, 0] = rows[:, 0].abs()
        rows[:10, 0] = 0
        rows[10:20] = -rows[:10]
    elif kind == "collinear" and width > 1:
        rows[:, -1] = rows[:, 0]
    rows /= rows.abs().amax(0).clamp_min(1)
    expected = expected_lp(rows)
    objective = rows.mean(0)
    if expected > 1e-8:
        with pytest.raises(KernelError) as caught:
            separation.certify_separation(replay(rows), objective, width)
        assert caught.value.code == "separation_detected"
    else:
        proof = separation.certify_separation(replay(rows), objective, width)
        assert 0 <= proof["separation_dual_upper_bound"] <= 1e-8
        assert proof["separation_peak_constraints"] <= max(8, 4 * width)


def test_exact_two_dimensional_vertex_oracle_independent_of_an_lp_solver():
    # Enumerate intersections of row faces and box faces, then evaluate every
    # feasible vertex. This oracle shares no numerical optimization code.
    a = np.array([[1., 2.], [2., -1.], [-1., 3.], [.5, 1.]])
    g = np.vstack([-a, np.eye(2), -np.eye(2)])
    h = np.r_[np.zeros(len(a)), np.ones(4)]
    c = a.mean(0)
    candidates = [np.zeros(2)]
    for pair in itertools.combinations(range(len(g)), 2):
        face = g[list(pair)]
        if abs(np.linalg.det(face)) > 1e-12:
            point = np.linalg.solve(face, h[list(pair)])
            if np.all(g @ point <= h + 1e-12):
                candidates.append(point)
    optimum = max(c @ point for point in candidates)
    normalized = a / np.max(np.abs(a), axis=1)[:, None]
    result = separation.solve_box_lp(torch.tensor(c), torch.tensor(normalized))
    assert c @ result.direction.numpy() == pytest.approx(optimum, abs=5e-10)
    assert result.upper_bound >= optimum - 2e-13
    assert result.upper_bound - optimum < 2e-9


@pytest.mark.parametrize("batch", [1, 7, 137])
@pytest.mark.parametrize("permutation", [False, True])
def test_quasi_separation_is_replay_and_permutation_invariant(batch, permutation):
    generator = torch.Generator().manual_seed(513)
    nuisance = torch.randn((137, 2), generator=generator, dtype=torch.float64)
    rows = torch.cat((torch.ones((137, 1), dtype=torch.float64), nuisance), 1)
    rows[:20, 0] = 0
    rows[20:40] = -rows[:20]
    if permutation:
        rows = rows[torch.randperm(len(rows), generator=generator)]
    with pytest.raises(KernelError) as caught:
        separation.certify_separation(replay(rows, batch), rows.mean(0), 3)
    assert caught.value.code == "separation_detected"


@pytest.mark.parametrize("scale", [1e-100, 1., 1e100])
@pytest.mark.parametrize("intercept", [False, True])
def test_dense_standardization_preserves_units_and_no_intercept(scale, intercept):
    values = torch.tensor([-2., -1., 0., 0., 1., 2.], dtype=torch.float64) * scale
    y = torch.tensor([0., 0., 0., 1., 1., 1.], dtype=torch.float64)
    x = torch.stack((torch.ones_like(values), values), 1) if intercept else values[:, None]
    with pytest.raises(AnalysisError) as caught:
        _check_binary_separation(y, x, intercept=intercept)
    assert caught.value.code == "separation_detected"


def test_arbitrary_positive_dual_gives_valid_bound_without_optimizer_convergence():
    rows = torch.tensor([[1., .4], [-1., -.2], [.3, 1.], [-.4, -1.]], dtype=torch.float64)
    objective = rows.mean(0)
    true_value = expected_lp(rows)
    for multipliers in ([0., 0., 0., 0.], [3., .1, 5., .2], [.2, .2, .3, .3]):
        upper = separation._upper_bound(objective, rows, torch.tensor(multipliers, dtype=torch.float64))
        assert upper >= true_value - 1e-13
    approximate = separation.solve_box_lp(objective, rows, max_iter=1)
    assert approximate.upper_bound >= true_value - 1e-13


def test_reported_optimizer_bound_is_not_trusted(monkeypatch):
    # A false cached success cannot bypass independently recomputed weak duality.
    rows = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    original = separation.solve_box_lp

    def misleading(objective, cuts):
        result = original(objective, cuts)
        result.upper_bound = 0.
        return result

    monkeypatch.setattr(separation, "solve_box_lp", misleading)
    with pytest.raises(KernelError) as caught:
        separation.certify_separation(replay(rows), rows.mean(0), 2)
    assert caught.value.code == "separation_detected"


def test_dual_multiplier_corruption_refuses_instead_of_accepting(monkeypatch):
    original = separation.solve_box_lp

    def corrupted(objective, rows):
        result = original(objective, rows)
        if len(rows):
            result.multipliers[:] = -1
        return result

    rows = torch.tensor([[1., .2], [-1., -.3], [.3, 1.], [-.4, -1.]], dtype=torch.float64)
    monkeypatch.setattr(separation, "solve_box_lp", corrupted)
    with pytest.raises(KernelError) as caught:
        separation.certify_separation(replay(rows), rows.mean(0), 2)
    assert caught.value.code == "separation_check_failed"


def test_pass_budget_refuses_inconclusive_program():
    rows = torch.tensor([[1., .2], [-1., -.3], [.3, 1.], [-.4, -1.]], dtype=torch.float64)
    with pytest.raises(KernelError) as caught:
        separation.certify_separation(replay(rows), rows.mean(0), 2, max_passes=1)
    assert caught.value.code == "separation_check_failed"


def test_workspace_guard_precedes_large_quadratic_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Workspace guard ran after allocation")
    monkeypatch.setattr(torch, "eye", forbidden)
    objective = torch.zeros(2000, dtype=torch.float64)
    with pytest.raises(KernelError) as caught:
        separation.certify_separation(lambda: (), objective, 2000)
    assert caught.value.code == "separation_check_failed" and "128 MiB" in str(caught.value)


@pytest.mark.parametrize("mode", ["nan", "wrong_shape", "wrong_precision"])
def test_malformed_witness_is_never_accepted(monkeypatch, mode):
    original = separation.solve_box_lp

    def corrupted(objective, rows):
        result = original(objective, rows)
        if mode == "nan":
            result.direction[:] = float("nan")
        elif mode == "wrong_shape":
            result.direction = result.direction[:-1]
        else:
            result.direction = result.direction.float()
        return result

    rows = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    monkeypatch.setattr(separation, "solve_box_lp", corrupted)
    with pytest.raises(KernelError) as caught:
        separation.certify_separation(replay(rows), rows.mean(0), 2)
    assert caught.value.code == "separation_check_failed"


def test_cpu_local_context_and_no_external_imports(monkeypatch):
    imported = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"scipy", "statsmodels", "linearmodels", "sklearn"}:
            raise AssertionError("External numerical import in native separation")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    rows = torch.tensor([[1., .2], [-1., -.3], [.3, 1.], [-.4, -1.]], dtype=torch.float64)
    with torch.device("meta"):
        result = separation.certify_separation(replay(rows), rows.mean(0), 2)
        assert torch.empty(0).device.type == "meta"
    assert result["separation_dual_upper_bound"] <= 1e-8
