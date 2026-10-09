"""Independent full-covariance MI contrast oracle and saved-state refusals."""
import json

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.stats import norm, t
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.joint import MIPoolResult, digest, mi_pool
from openecon.econometrics.mi.lincom import MILincomResult, mi_lincom
from openecon.resources import use_workspace_budget


def inputs(m=8, p=3, complete_df=60, alpha=0.05):
    rng = np.random.default_rng(78104)
    q = rng.normal(size=(m, p)) + np.linspace(-0.5, 2, p)
    covariance = []
    for _ in range(m):
        a = rng.normal(size=(p, p))
        covariance.append(a @ a.T + np.eye(p))
    u = np.array(covariance)
    pool = mi_pool(q, u, terms=[f"x{j}" for j in range(p)],
                   imputation_ids=[f"draw-{j + 1}" for j in range(m)],
                   complete_df=complete_df, alpha=alpha,
                   imputation_description="Declared common estimands under eight justified imputations")
    return q, u, pool


def oracle(q, u, r, null, complete_df, alpha):
    """NumPy moments and SciPy tails, without implementation intermediates."""
    estimates = np.einsum("mp,kp->mk", q, r)
    covariance = np.array([r @ s @ r.T for s in u])
    m = len(q)
    mean = np.mean(estimates, axis=0)
    within = np.mean(covariance, axis=0)
    centered = estimates - mean
    between = centered.T @ centered / (m - 1)
    total = within + (1 + 1 / m) * between
    added = (1 + 1 / m) * between.diagonal()
    lam = added / total.diagonal()
    riv = added / within.diagonal()
    with np.errstate(divide="ignore"):
        old = (m - 1) / lam**2
    if complete_df is None:
        df = old
    else:
        observed = complete_df * (complete_df + 1) / (complete_df + 3) * (1 - lam)
        df = 1 / (1 / old + 1 / observed)
    se = np.sqrt(total.diagonal())
    statistic = (mean - null) / se
    critical = t.isf(alpha / 2, df)
    return {
        "estimate": mean, "std_error": se, "statistic": statistic, "df": df,
        "p_value": 2 * t.sf(np.abs(statistic), df),
        "ci_low": mean - critical * se, "ci_high": mean + critical * se,
        "lambda": lam, "relative_increase_variance": riv,
        "fraction_missing_information": (riv + 2 / (df + 3)) / (1 + riv),
    }, estimates, covariance, within, between, total


@pytest.mark.parametrize("complete_df", [None, 40.0])
@pytest.mark.parametrize("alpha", [0.05, 0.1])
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("nonzero_null", [False, True])
def test_full_contrast_numpy_scipy_oracle(complete_df, alpha, k, nonzero_null):
    q, u, pool = inputs(complete_df=complete_df)
    r = np.array([[1.0, 0.4, -0.8], [0.0, 1.0, 0.3]])[:k]
    null = np.linspace(-0.6, 0.7, k) if nonzero_null else np.zeros(k)
    result = mi_lincom(pool, r, values=null, names=[f"contrast {j}" for j in range(k)], alpha=alpha)
    expected, eq, eu, within, between, total = oracle(q, u, r, null, complete_df, alpha)
    for field, value in expected.items():
        assert_allclose(result.table[field].astype(float), value, rtol=4e-12, atol=2e-13)
    assert_allclose(result.transformed_pool.estimates, eq, rtol=1e-13, atol=1e-13)
    assert_allclose(result.transformed_pool.covariances, eu, rtol=1e-13, atol=1e-13)
    assert_allclose(result.tables["within_covariance"], within, rtol=1e-13)
    assert_allclose(result.tables["between_covariance"], between, rtol=1e-13)
    assert_allclose(result.tables["total_covariance"], total, rtol=1e-13)
    assert_allclose(result.table["null_value"], null)
    assert result.transformed_pool.imputation_ids == pool.imputation_ids
    assert result.transformed_pool.imputation_description == pool.imputation_description
    assert result.source_pool == pool
    assert result.metadata["source_pool_integrity_sha256"] == pool.integrity_sha256
    assert result.metadata["multiple_testing_adjustment"] == "none"
    assert result.metadata["joint_test"] is False
    # This fixture requires off-diagonal uncertainty to obtain correct variances.
    diagonal_only = np.array([r @ np.diag(s.diagonal()) @ r.T for s in u])
    assert not np.allclose(eu[:, 0, 0], diagonal_only[:, 0, 0])


def test_zero_between_variance_normal_and_finite_complete_df_limits():
    q = [[1.0, 2.0, -1.0]] * 8
    u = [[[2.0, 0.8, 0.1], [0.8, 3.0, -0.2], [0.1, -0.2, 4.0]]] * 8
    r = [[1.0, -0.5, 0.2]]
    null = [0.4]
    for complete_df in (None, 60):
        source = mi_pool(q, u, complete_df=complete_df,
                         imputation_description="Identical common estimates with zero between variance")
        result = mi_lincom(source, r, values=null)
        expected, *_ = oracle(np.array(q), np.array(u), np.array(r), null, complete_df, 0.05)
        assert_allclose(result.table.std_error, expected["std_error"])
        assert_allclose(result.table.p_value, expected["p_value"], rtol=4e-12)
        assert_allclose(result.table.ci_low, expected["ci_low"], rtol=4e-12)
        if complete_df is None:
            assert result.table.df.iloc[0] is None
            assert_allclose(result.table.p_value, 2 * norm.sf(abs(expected["statistic"])))
        else:
            assert_allclose(result.table.df, [60 * 61 / 63])


def test_large_nonzero_null_cannot_erase_between_variance_or_move_intervals():
    _, _, source = inputs()
    ordinary = mi_lincom(source, [[1.0, 0.3, -0.1]])
    large_null = mi_lincom(source, [[1.0, 0.3, -0.1]], values=[1e20])
    for field in ("estimate", "std_error", "df", "lambda", "fraction_missing_information", "ci_low", "ci_high"):
        assert ordinary.table[field].equals(large_null.table[field])
    assert ordinary.transformed_pool.estimates == large_null.transformed_pool.estimates
    assert ordinary.transformed_pool.covariances == large_null.transformed_pool.covariances
    assert large_null.table.statistic.iloc[0] < -1e18
    assert large_null.table.p_value.iloc[0] < 1e-30


def test_default_alpha_override_and_declared_null_equal_estimate():
    _, _, source = inputs(alpha=0.1)
    default = mi_lincom(source, [[0.0, 1.0, 0.0]])
    assert default.alpha == 0.1
    assert default.names == ("contrast-1",)
    estimate = default.table.estimate.iloc[0]
    at_null = mi_lincom(source, [[0.0, 1.0, 0.0]], values=[estimate], alpha=0.01)
    assert_allclose(at_null.table.statistic, [0])
    assert_allclose(at_null.table.p_value, [1])
    assert at_null.table.ci_high.iloc[0] > default.table.ci_high.iloc[0]


def test_json_replay_frozen_nested_state_validated_copy_and_rehashed_scientific_refusal():
    _, _, source = inputs()
    result = mi_lincom(source, [[1.0, -0.3, 0.2], [0.0, 1.0, 0.4]], values=[0.7, -0.1])
    restored = MILincomResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.table.equals(result.table)
    assert restored.latex == result.latex
    assert "tabular" in restored.to_latex()
    assert result.model_copy(deep=True) == result
    with pytest.raises(ValueError):
        result.model_copy(update={"values": (1.0, 2.0)})
    with pytest.raises(ValueError):
        result.values = (1.0, 2.0)
    metadata = result.metadata
    metadata["workspace"]["budget_bytes"] = 0
    assert result.metadata["workspace"]["budget_bytes"] > 0
    # A recomputed checksum is not authorization to alter the projection law.
    state = json.loads(result.model_dump_json())
    state["transformed_pool"]["estimates"][0][0] += 0.01
    nested = state["transformed_pool"]
    nested["integrity_sha256"] = digest({k: v for k, v in nested.items() if k != "integrity_sha256"})
    state["integrity_sha256"] = digest({k: v for k, v in state.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="projection"):
        MILincomResult.model_validate(state)
    # A changed inference contract with its own valid checksum is still refused.
    state = json.loads(result.model_dump_json())
    state["transformed_pool"]["complete_df"] = 30.0
    nested = state["transformed_pool"]
    nested["integrity_sha256"] = digest({k: v for k, v in nested.items() if k != "integrity_sha256"})
    state["integrity_sha256"] = digest({k: v for k, v in state.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="contract"):
        MILincomResult.model_validate(state)


@pytest.mark.parametrize("r", [
    None, [], [1.0, 0.0, 0.0], [[1.0, 0.0]], np.eye(4),
    [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    [[True, 0.0, 0.0]], np.ones((1, 3), dtype=bool),
    np.ones((1, 3), dtype=complex), np.ones((1, 3), dtype=object),
    [[float("nan"), 1.0, 0.0]], [[float("inf"), 0.0, 0.0]],
    [[1e-300, 0.0, 0.0]], [[1e200, 0.0, 0.0]],
])
def test_invalid_contrasts_are_refused(r):
    _, _, source = inputs()
    with pytest.raises((AnalysisError, ValueError)):
        mi_lincom(source, r)


@pytest.mark.parametrize("options", [
    {"values": [True]}, {"values": [1 + 2j]}, {"values": [float("nan")]},
    {"values": [float("inf")]}, {"values": []}, {"values": [0.0, 1.0]},
    {"values": np.ones(1, dtype=object)}, {"names": [True]}, {"names": [""]},
    {"names": ["a" * 257]}, {"names": "term"}, {"alpha": True},
    {"alpha": float("nan")}, {"alpha": 0}, {"max_work": True},
    {"max_work": 0}, {"max_work": 1.5}, {"max_work": 1_000_000_001},
])
def test_invalid_values_labels_alpha_and_work_limits(options):
    _, _, source = inputs()
    with pytest.raises((AnalysisError, ValueError)):
        mi_lincom(source, [[1.0, 0.0, 0.0]], **options)


def test_duplicate_labels_source_hash_and_unsupported_pool_are_refused():
    _, _, source = inputs()
    with pytest.raises(AnalysisError):
        mi_lincom(source, [[1, 0, 0], [0, 1, 0]], names=["same", "same"])
    with pytest.raises(AnalysisError):
        mi_lincom(source.model_dump(mode="json"), [[1, 0, 0]])
    state = source.model_dump(mode="json")
    state["estimates"][0][0] += 0.01
    bypass = MIPoolResult.model_construct(**state)
    with pytest.raises(ValueError, match="integrity"):
        mi_lincom(bypass, [[1, 0, 0]])


def test_no_workspace_work_or_saved_restore_buffer_bypass(monkeypatch):
    _, _, source = inputs(p=16)
    result = mi_lincom(source, np.eye(16))
    saved = result.model_dump(mode="json")
    def allocation_forbidden(*args, **kwargs):
        raise AssertionError("Tensor coercion preceded resource refusal")
    with monkeypatch.context() as patch:
        patch.setattr(torch, "as_tensor", allocation_forbidden)
        with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
            mi_lincom(source, np.eye(16))
        with use_workspace_budget(1), pytest.raises((AnalysisError, ValueError), match="workspace"):
            MILincomResult.model_validate(saved)
        with pytest.raises(AnalysisError, match="max_work"):
            mi_lincom(source, np.eye(16), max_work=1)
    # A larger current resource budget does not change the saved original plan.
    with use_workspace_budget(1024):
        restored = MILincomResult.model_validate(saved)
    assert restored.model_dump_json() == result.model_dump_json()


@pytest.mark.parametrize("field", ["weights", "values", "max_work", "work_estimate"])
def test_raw_saved_boolean_coercion_cannot_pass_rehashed_state(field):
    _, _, source = inputs()
    result = mi_lincom(source, [[1.0, 0.0, 0.0]])
    state = result.model_dump(mode="json")
    if field == "weights":
        state[field][0][0] = True
    elif field == "values":
        state[field][0] = True
    else:
        state[field] = True
    state["integrity_sha256"] = digest({k: v for k, v in state.items() if k != "integrity_sha256"})
    with pytest.raises((AnalysisError, ValueError)):
        MILincomResult.model_validate(state)


def test_metadata_envelopes_and_saved_projected_dimensions_precede_buffers(monkeypatch):
    _, _, source = inputs()
    result = mi_lincom(source, [[1.0, 0.0, 0.0]])
    def allocation_forbidden(*args, **kwargs):
        raise AssertionError("Tensor coercion preceded envelope refusal")
    with monkeypatch.context() as patch:
        patch.setattr(torch, "as_tensor", allocation_forbidden)
        raw = source.model_dump(mode="json")
        raw["metadata_json"] = " " * 131073
        with pytest.raises(AnalysisError, match="string budget"):
            mi_lincom(MIPoolResult.model_construct(**raw), [[1, 0, 0]])
        state = result.model_dump(mode="json")
        state["metadata_json"] = " " * 131073
        with pytest.raises((AnalysisError, ValueError), match="string budget"):
            MILincomResult.model_validate(state)
        state = result.model_dump(mode="json")
        state["transformed_pool"] = source.model_dump(mode="json")
        with pytest.raises((AnalysisError, ValueError), match="dimensions"):
            MILincomResult.model_validate(state)


def test_metadata_contract_and_integrity_tampering_are_refused():
    _, _, source = inputs()
    result = mi_lincom(source, [[1, 0, 0]])
    for change in ("stata_parity_validated", "multiple_testing_adjustment", "workspace"):
        state = result.model_dump(mode="json")
        metadata = json.loads(state["metadata_json"])
        if change == "workspace":
            metadata[change]["estimated_workspace_bytes"] += 1
        else:
            metadata[change] = True
        state["metadata_json"] = json.dumps(metadata)
        state["integrity_sha256"] = digest({k: v for k, v in state.items() if k != "integrity_sha256"})
        with pytest.raises(ValueError, match="contracts"):
            MILincomResult.model_validate(state)
    state = result.model_dump(mode="json")
    state["values"][0] = 1.0
    with pytest.raises(ValueError, match="integrity"):
        MILincomResult.model_validate(state)


def test_cpu_double_and_local_operations_ignore_ambient_device_dtype_and_rng():
    _, _, source = inputs()
    torch.manual_seed(814)
    before = torch.get_rng_state().clone()
    old_dtype = torch.get_default_dtype()
    torch.set_default_device("meta")
    torch.set_default_dtype(torch.float32)
    try:
        result = mi_lincom(source, [[1.0, -0.5, 0.2]], values=[0.3])
        assert result.table.p_value.iloc[0] > 0
        assert result.metadata["precision"] == "float64"
        assert result.metadata["device"] == "cpu"
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(old_dtype)
    assert torch.equal(before, torch.get_rng_state())
    accelerator = torch.ones((1, 3), device="meta")
    with pytest.raises(AnalysisError, match="CPU"):
        mi_lincom(source, accelerator)
