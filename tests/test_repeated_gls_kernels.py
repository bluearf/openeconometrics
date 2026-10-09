"""Kernel invariants separate from the independent R/nlme scientific oracles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mixed import repeated_kernels as kernels


CASES = json.loads((Path(__file__).parent / "fixtures/repeated_gls/nlme.json").read_text())["cases"]
BUDGET = 2_000_000_000


def sample(name):
    case = CASES[name]
    data = case["data"]
    subjects = list(dict.fromkeys(row["subject"] for row in data))
    levels = sorted({row["occasion"] for row in data})
    x = torch.tensor([[1, row["x"], row["z"]] for row in data], dtype=torch.float64)
    y = torch.tensor([row["y"] for row in data], dtype=torch.float64)
    groups = [subjects.index(row["subject"]) for row in data]
    occasions = [levels.index(row["occasion"]) for row in data]
    structure, method = name.split("_")
    return x, y, groups, occasions, levels, structure, method.upper()


def array(value):
    return torch.tensor(value, dtype=torch.float64)


@pytest.fixture(scope="module")
def fitted():
    return {name: kernels.fit(*sample(name), BUDGET) for name in CASES}


@pytest.mark.parametrize("name", CASES)
def test_replay_recomputes_every_scientific_key_without_optimizer(name, fitted, monkeypatch):
    original = fitted[name]
    monkeypatch.setattr(kernels, "maximize", lambda *args: pytest.fail("replay invoked an optimizer"))
    restored = kernels.replay(*sample(name), original["theta"], BUDGET)
    assert set(restored) == set(original) - {"optimizer"}
    for key, value in restored.items():
        if isinstance(value, str) or value is None or key == "covariance_terms":
            assert value == original[key]
        elif isinstance(value, dict):
            assert value.keys() == original[key].keys()
            for item in value:
                torch.testing.assert_close(array(value[item]), array(original[key][item]), rtol=1e-10, atol=1e-11)
        else:
            torch.testing.assert_close(array(value), array(original[key]), rtol=1e-10, atol=1e-11)
    assert original["optimizer"]["converged"]
    assert 0 < original["optimizer"]["work_used"] <= BUDGET


@pytest.mark.parametrize("structure", ["cs", "ar1", "diagonal", "unstructured"])
def test_ml_full_joint_blocks_and_natural_parameter_delta_transform(structure, fitted):
    name = structure + "_ml"
    result = fitted[name]
    geometry = kernels._Geometry(*sample(name), BUDGET)
    raw = array(result["theta"])
    jacobian = torch.autograd.functional.jacobian(geometry.parameter_report, raw)
    transform = torch.block_diag(torch.eye(geometry.p, dtype=torch.float64), jacobian)
    joint_raw = array(result["joint_raw_covariance"])
    joint = array(result["joint_covariance"])
    torch.testing.assert_close(joint, transform @ joint_raw @ transform.T)
    torch.testing.assert_close(joint_raw[:3, :3], array(result["covariance"]))
    torch.testing.assert_close(joint_raw[3:, 3:], array(result["theta_covariance"]))
    torch.testing.assert_close(joint_raw[:3, 3:], array(result["beta_theta_covariance"]))
    torch.testing.assert_close(joint[3:, 3:], array(result["covariance_parameter_covariance"]))
    assert float(joint_raw[:3, 3:].abs().max()) > 1e-6
    assert float((array(result["covariance"]) - array(result["conditional_gls_covariance"])).abs().max()) > 1e-6
    assert float(torch.linalg.eigvalsh(joint_raw)[0]) > 0


@pytest.mark.parametrize("structure", ["cs", "ar1", "diagonal", "unstructured"])
def test_reml_keeps_separate_profile_covariance_and_conditional_fixed_inference(structure, fitted):
    result = fitted[structure + "_reml"]
    assert result["joint_covariance"] is None
    assert result["joint_raw_covariance"] is None
    assert result["beta_theta_covariance"] is None
    assert result["covariance"] == result["conditional_gls_covariance"]
    assert "no KR/Satterthwaite" in result["inference"]
    assert float(torch.linalg.eigvalsh(array(result["theta_covariance"]))[0]) > 0


@pytest.mark.parametrize("name", CASES)
def test_nonsingular_fixed_design_coordinates_preserve_likelihood_and_full_uncertainty(name, fitted):
    original = fitted[name]
    x, y, groups, occasions, levels, structure, method = sample(name)
    transform = array([[2., .3, -.2], [0., -.5, .1], [0., .2, 3.]])
    inverse = torch.linalg.inv(transform)
    new = kernels.replay(x @ transform, y, groups, occasions, levels, structure, method, original["theta"], BUDGET)
    torch.testing.assert_close(array(new["beta"]), inverse @ array(original["beta"]), rtol=2e-9, atol=1e-10)
    torch.testing.assert_close(array(new["covariance"]), inverse @ array(original["covariance"]) @ inverse.T, rtol=2e-8, atol=1e-10)
    torch.testing.assert_close(array(new["theta_covariance"]), array(original["theta_covariance"]), rtol=2e-8, atol=1e-10)
    assert new["loglik_ml"] == pytest.approx(original["loglik_ml"], abs=2e-9)
    assert new["loglik_reml"] == pytest.approx(original["loglik_reml"], abs=2e-9)
    assert new["reml_normalization"] == "logdet(X'V^-1X) minus logdet(X'X)"
    if method == "ML":
        whole = torch.block_diag(inverse, torch.eye(len(original["theta"]), dtype=torch.float64))
        torch.testing.assert_close(array(new["joint_raw_covariance"]), whole @ array(original["joint_raw_covariance"]) @ whole.T, rtol=2e-8, atol=1e-10)


@pytest.mark.parametrize("name", CASES)
def test_occasion_origin_translation_changes_no_geometry_or_signed_integer_gap(name, fitted):
    original = fitted[name]
    x, y, groups, occasions, levels, structure, method = sample(name)
    new = kernels.replay(x, y, groups, occasions, [level - 100 for level in levels], structure, method, original["theta"], BUDGET)
    for key in ("beta", "covariance", "covariance_parameters", "covariance_parameter_covariance", "occasion_covariance", "residual_covariance"):
        torch.testing.assert_close(array(new[key]), array(original[key]), rtol=1e-12, atol=1e-12)
    if structure == "ar1":
        rho = new["covariance_parameters"][1]
        assert rho < 0
        covariance = array(new["occasion_covariance"])
        assert float(covariance[0, 1]) < 0
        assert float(covariance[1, 2]) > 0  # actual gap two, not compressed occasion index


@pytest.mark.parametrize("name", CASES)
def test_replay_rejects_nonstationary_resealed_parameter_point(name, fitted):
    theta = array(fitted[name]["theta"])
    theta[0] += .15
    with pytest.raises(AnalysisError, match="stationary"):
        kernels.replay(*sample(name), theta, BUDGET)


def test_kernel_refuses_signed_ar1_alias_grid_and_unidentified_unstructured_pairs():
    x, y, groups, occasions, levels, _, _ = sample("ar1_ml")
    with pytest.raises(AnalysisError, match="greatest common"):
        kernels.fit(x, y, groups, occasions, [2 * level for level in levels], "ar1", "ML", BUDGET)
    keep = array([not (occasion == 3 and group > 1) for occasion, group in zip(occasions, groups)]).bool()
    with pytest.raises(AnalysisError, match="four subjects"):
        kernels.fit(x[keep], y[keep], array(groups).long()[keep], array(occasions).long()[keep], levels, "unstructured", "ML", BUDGET)


@pytest.mark.parametrize("budget", [True, 0, 2_000_000_001])
def test_kernel_refuses_invalid_work_budget(budget):
    with pytest.raises(AnalysisError, match="max_work"):
        kernels.fit(*sample("cs_ml"), budget)


def test_fit_and_replay_charge_derivatives_to_explicit_budget(fitted):
    with pytest.raises(AnalysisError, match="work"):
        kernels.fit(*sample("cs_ml"), 1)
    with pytest.raises(AnalysisError, match="work"):
        kernels.replay(*sample("cs_ml"), fitted["cs_ml"]["theta"], 1)


def test_duplicate_subject_occasion_and_rank_deficiency_refused():
    x, y, groups, occasions, levels, structure, method = sample("cs_ml")
    duplicate = occasions.copy()
    duplicate[1] = duplicate[0]
    with pytest.raises(AnalysisError, match="duplicate"):
        kernels.fit(x, y, groups, duplicate, levels, structure, method, BUDGET)
    x[:, 2] = x[:, 1]
    with pytest.raises(AnalysisError, match="rank deficient"):
        kernels.fit(x, y, groups, occasions, levels, structure, method, BUDGET)


def test_kernel_refuses_non_cpu_double_and_invalid_saved_theta():
    x, y, groups, occasions, levels, structure, method = sample("cs_ml")
    with pytest.raises(AnalysisError, match="CPU float64"):
        kernels.fit(x.float(), y, groups, occasions, levels, structure, method, BUDGET)
    with pytest.raises(AnalysisError, match="dimensions"):
        kernels.replay(x, y, groups, occasions, levels, structure, method, [0.], BUDGET)
    with pytest.raises(AnalysisError, match="dimensions"):
        kernels.replay(x, y, groups, occasions, levels, structure, method, [0., float("nan")], BUDGET)


def test_signed_ar1_independence_is_an_interior_fit_with_exact_finite_joint_information():
    # Orthogonal centered Hadamard columns yield exact empirical covariance I.
    # Zero rho is an interior point, not a residual-covariance boundary.
    hadamard = torch.ones((1, 1), dtype=torch.float64)
    for _ in range(4):
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1),
                              torch.cat((hadamard, -hadamard), 1)), 0)
    y = hadamard[:, 1:5].flatten()
    x = torch.ones((len(y), 1), dtype=torch.float64)
    groups = [row // 4 for row in range(len(y))]
    occasions = [row % 4 for row in range(len(y))]
    result = kernels.fit(x, y, groups, occasions, [0, 1, 3, 4], "ar1", "ML", BUDGET)
    torch.testing.assert_close(array(result["covariance_parameters"]), array([1., 0.]), rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(array(result["beta"]), array([0.]), atol=1e-12, rtol=0)
    torch.testing.assert_close(array(result["joint_raw_covariance"]),
                               torch.diag(array([1 / 64, 1 / 128, 1 / 32])), atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(array(result["covariance_parameter_covariance"]),
                               torch.diag(array([1 / 32, 1 / 32])), atol=1e-10, rtol=1e-9)
    restored = kernels.replay(x, y, groups, occasions, [0, 1, 3, 4], "ar1", "ML", result["theta"], BUDGET)
    torch.testing.assert_close(array(restored["joint_raw_covariance"]), array(result["joint_raw_covariance"]), atol=1e-12, rtol=1e-12)
