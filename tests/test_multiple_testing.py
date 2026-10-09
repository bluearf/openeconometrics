"""Independent adjustment references, joint-null enumeration and error control."""

import json

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests as reference
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


@pytest.mark.parametrize(
    "method,ref",
    [
        ("bonferroni", "bonferroni"),
        ("sidak", "sidak"),
        ("holm", "holm"),
        ("holm_sidak", "holm-sidak"),
        ("hochberg", "simes-hochberg"),
        ("hommel", "hommel"),
        ("bh", "fdr_bh"),
        ("by", "fdr_by"),
    ],
)
def test_independent_adjustment_reference_and_permutation(method, ref):
    rng = np.random.default_rng(8251)
    for n in (1, 2, 11, 83):
        p = np.r_[0, 1, 0.05, 0.05, rng.uniform(size=n)]
        perm = rng.permutation(len(p))
        expected_reject, expected_p, *_ = reference(p, method=ref)
        actual = oe.multipletests(p, method=method)
        assert_allclose(actual.adjusted_p_value, expected_p, atol=3e-15)
        assert list(actual.reject) == list(expected_reject)
        changed = oe.multipletests(p[perm], method=method)
        assert_allclose(changed.adjusted_p_value, expected_p[perm], atol=3e-15)
        exported = actual.to_latex()
        assert "\\end{tabular}" in exported or "\\end{longtable}" in exported
        json.dumps(actual.attrs, allow_nan=False)


def test_missing_and_family_contract():
    with pytest.raises(AnalysisError, match="missing"):
        oe.multipletests([0.01, None, 0.2])
    actual = oe.multipletests([0.01, None, 0.2], missing="drop", labels=["a", "b", "c"])
    assert list(actual.hypothesis) == ["a", "b", "c"]
    assert actual.attrs["tested_family_size"] == 2 and actual.attrs["excluded_missing"] == 1
    assert actual.adjusted_p_value.iloc[0] == 0.02 and np.isnan(actual.adjusted_p_value.iloc[1])
    for value in ([], [True], [1.01], [-0.01], [np.inf], [np.nan], [complex(0.1, 0.2)]):
        with pytest.raises(AnalysisError):
            oe.multipletests(value)
    with pytest.raises(AnalysisError):
        oe.multipletests([0.1, 0.2], labels=["same", "same"])


def test_romano_wolf_matches_exhaustive_joint_null_and_tied_ordering():
    from itertools import product

    design = np.array([[1.0, 2.0, -0.5], [-1.0, 0.5, 2.0], [0.8, -1.0, 0.3]])
    signs = np.array(list(product((-1, 1), repeat=3)))
    null = signs @ design
    observed = np.array([3.0, 3.0, 1.0])
    result = oe.stepdown(
        observed,
        null,
        calibration="enumerated",
        null_description="All eight equiprobable sign assignments",
    )
    expected_first = np.mean(np.max(np.abs(null), axis=1) >= 3)
    expected_last = max(expected_first, np.mean(np.abs(null[:, 2]) >= 1))
    assert_allclose(result.adjusted_p_value, [expected_first, expected_first, expected_last])
    perm = [1, 2, 0]
    changed = oe.stepdown(
        observed[perm],
        null[:, perm],
        calibration="enumerated",
        null_description="Same complete sign assignments",
    )
    assert_allclose(changed.adjusted_p_value, np.asarray(result.adjusted_p_value)[perm])
    sampled = oe.stepdown(observed, null, null_description="Monte Carlo sign sample")
    assert sampled.adjusted_p_value.iloc[0] == (1 + 8 * expected_first) / 9


def test_westfall_young_minp_joint_reference():
    null = [[0.1, 0.2], [0.01, 0.3], [0.5, 0.06], [0.4, 0.4]]
    actual = oe.stepdown(
        [0.05, 0.2],
        null,
        method="westfall_young",
        calibration="enumerated",
        null_description="Complete permutation null p-value matrix",
    )
    assert_allclose(actual.adjusted_p_value, [0.25, 0.5])
    with pytest.raises(AnalysisError):
        oe.stepdown([0.05], [[0.2], [0.3]])


def test_normal_simultaneous_intervals_correlations_and_rng():
    from scipy.stats import norm

    before = torch.random.get_rng_state().clone()
    actual = oe.simultaneous_ci(
        [1.0, 2.0],
        [[4.0, 6.0], [6.0, 9.0]],
        draws=50000,
        seed=317,
        family_description="Perfectly correlated normal estimators with known joint covariance",
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert abs(actual.attrs["critical_value"] - norm.ppf(0.975)) < 0.025
    assert_allclose(
        (actual.ci_high - actual.estimate) / actual.std_error, actual.attrs["critical_value"]
    )
    independent = oe.simultaneous_ci(
        [0.0, 0.0],
        np.eye(2),
        family_description="Independent known-variance normal means",
        seed=317,
    )
    assert abs(independent.attrs["critical_value"] - norm.ppf((1 + np.sqrt(0.95)) / 2)) < 0.025
    assert independent.attrs["critical_value"] > actual.attrs["critical_value"]
    for covariance in (
        [[1.0, 2.0], [2.0, 1.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0]],
    ):
        with pytest.raises(AnalysisError):
            oe.simultaneous_ci(
                [0.0, 0.0], covariance, family_description="Invalid joint covariance"
            )


def test_bonferroni_global_null_size_independent_and_dependent():
    from scipy.stats import norm

    rng = np.random.default_rng(5643)
    for correlation in (0.0, 0.8):
        common = rng.normal(size=(1000, 1))
        noise = np.sqrt(correlation) * common + np.sqrt(1 - correlation) * rng.normal(
            size=(1000, 12)
        )
        p = 2 * norm.sf(np.abs(noise))
        # Exact Bonferroni threshold is an independent family-level oracle.
        expected = np.any(p <= 0.05 / 12, axis=1)
        actual = [oe.multipletests(row, method="bonferroni").reject.any() for row in p]
        assert list(expected) == actual
        assert np.mean(actual) < 0.09


def _closed_simes(p):
    """Original closed-testing definition, independent of Hommel's shortcut."""
    from itertools import combinations

    output = [0.0] * len(p)
    for size in range(1, len(p) + 1):
        for subset in combinations(range(len(p)), size):
            ordered = sorted(p[i] for i in subset)
            value = min(1.0, min(size * x / rank for rank, x in enumerate(ordered, 1)))
            for index in subset:
                output[index] = max(output[index], value)
    return output


@pytest.mark.parametrize(
    "method", ["bonferroni", "sidak", "holm", "holm_sidak", "hochberg", "hommel", "bh", "by"]
)
def test_singleton_boundary_monotonicity_ties_and_full_family_state(method):
    for p in (0.0, 0.001, 0.05, 0.5, 1.0):
        assert_allclose(oe.multipletests([p], method=method).adjusted_p_value, [p], atol=1e-15)
    p = np.array([1.0, 0.1, 0.01, 0.1, 0.0, 0.3])
    result = oe.multipletests(p, method=method, labels=list("abcdef"))
    sorted_adjusted = result.adjusted_p_value.iloc[np.argsort(p)].to_numpy()
    assert np.all(np.diff(sorted_adjusted) >= -1e-15)
    assert result.adjusted_p_value.iloc[1] == result.adjusted_p_value.iloc[3]
    assert np.all(result.adjusted_p_value >= p - 1e-15)
    assert result.attrs["family_members"] == list("abcdef")
    assert result.attrs["ordering_positions"] == [4, 2, 1, 3, 5, 0]
    assert result.attrs["tie_groups"] == [[4], [2], [1, 3], [5], [0]]
    assert result.attrs["rejection_rule"] == "adjusted_p_value <= alpha"
    if method == "hommel":
        assert_allclose(result.adjusted_p_value, _closed_simes(p), atol=1e-15)


def test_hommel_independent_closed_testing_exhaustive_small_families():
    from itertools import product

    for p in product((0.0, 0.01, 0.15, 1.0), repeat=4):
        assert_allclose(
            oe.multipletests(p, method="hommel").adjusted_p_value, _closed_simes(p), atol=1e-15
        )


class NeverRead:
    """Budget checks must use shape/length before accessing payload or copying."""

    def __init__(self, shape):
        self.shape = shape

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        raise AssertionError("payload read before the budget check")

    def __array__(self, *args, **kwargs):
        raise AssertionError("array copy before the budget check")


@pytest.mark.parametrize("missing", ["drop", "raise"])
def test_raw_input_budget_precedes_missing_scan_and_copy(missing):
    with pytest.raises(AnalysisError, match="budget"):
        oe.multipletests(NeverRead((100001,)), missing=missing)
    with pytest.raises(AnalysisError, match="budget"):
        oe.multipletests(NeverRead((5001,)), method="hommel", missing=missing)


def test_joint_matrix_work_guard_precedes_copy():
    with pytest.raises(AnalysisError, match="budget"):
        oe.stepdown([1.0, 2.0], NeverRead((4000001, 2)), null_description="Unreached input")
    with pytest.raises(AnalysisError, match="budget"):
        oe.stepdown([1.0] * 400, NeverRead((626, 400)), null_description="Unreached input")
    with pytest.raises(AnalysisError, match="budget"):
        oe.simultaneous_ci(
            NeverRead((385,)), NeverRead((385, 385)), family_description="Unreached input"
        )


@pytest.mark.parametrize(
    "values",
    [
        [True, 0.2],
        [np.bool_(True)],
        [".1", ".2"],
        [0.1, 1j],
        [[0.1]],
        (x for x in [0.1]),
        {"a": 0.1},
        [pd.NA],
    ],
)
def test_numeric_vectors_do_not_coerce_invalid_types(values):
    with pytest.raises(AnalysisError):
        oe.multipletests(values)


@pytest.mark.parametrize(
    "matrix",
    [
        [[True, False], [False, True]],
        [["1", 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0]],
        [[1j, 0.0], [0.0, 1.0]],
        [[1.0, np.inf], [0.0, 1.0]],
    ],
)
def test_numeric_matrix_strictness(matrix):
    with pytest.raises(AnalysisError):
        oe.stepdown([1.0, 2.0], matrix, null_description="Invalid draws")
    with pytest.raises(AnalysisError):
        oe.simultaneous_ci([1.0, 2.0], matrix, family_description="Invalid covariance")


def test_nullable_missing_keeps_original_positions_and_correct_family():
    import pandas as pd

    result = oe.multipletests(pd.Series([0.1, pd.NA, 0.01], dtype="Float64"), missing="drop")
    assert result.attrs["tested_positions"] == [0, 2]
    assert result.attrs["excluded_positions"] == [1]
    assert result.attrs["ordering_positions"] == [2, 0]
    assert_allclose(result.adjusted_p_value.iloc[[0, 2]], [0.1, 0.02])
    assert result.reject.iloc[1] is None


def test_westfall_young_preserves_supplied_raw_p_values():
    result = oe.stepdown(
        [0.05, 0.2],
        [[0.1, 0.2], [0.01, 0.3], [0.5, 0.06], [0.4, 0.4]],
        method="westfall_young",
        calibration="enumerated",
        null_description="Complete joint marginal p-values",
    )
    assert result.p_value.tolist() == [0.05, 0.2]
    assert result.calibrated_marginal_p_value.tolist() == [0.25, 0.5]
    assert result.attrs["statistic_kind"] == "supplied marginal p-value"


@pytest.mark.parametrize("tail", ["two-sided", "greater", "less"])
def test_stepdown_ties_permutation_and_independent_tail_counts(tail):
    observed = np.array([-3.0, 3.0, 1.0, -3.0])
    null = np.array([[2.0, 4.0, -1.0, -2.0], [-4.0, -1.0, 1.0, 4.0], [1.0, 2.0, -2.0, -1.0]])
    transformed = (
        np.abs(observed) if tail == "two-sided" else (-observed if tail == "less" else observed)
    )
    transformed_null = np.abs(null) if tail == "two-sided" else (-null if tail == "less" else null)
    order = sorted(range(4), key=lambda i: -transformed[i])
    expected, previous = [0.0] * 4, 0.0
    for cutoff in sorted(set(transformed), reverse=True):
        remaining = [i for i in order if transformed[i] <= cutoff]
        count = sum(max(row[i] for i in remaining) >= cutoff for row in transformed_null)
        previous = max(previous, (count + 1) / 4)
        for i in remaining:
            if transformed[i] == cutoff:
                expected[i] = previous
    actual = oe.stepdown(
        observed, null, tail=tail, null_description="Independent Python count oracle"
    )
    assert_allclose(actual.adjusted_p_value, expected)
    for perm in ([3, 2, 0, 1], [1, 0, 3, 2]):
        changed = oe.stepdown(
            observed[perm], null[:, perm], tail=tail, null_description="Same permuted joint law"
        )
        assert_allclose(changed.adjusted_p_value, np.array(expected)[perm])


def test_covariance_symmetry_is_scale_invariant_and_full_joint_state_retained():
    for scale in (1.0, 1e-20, 1e200):
        with pytest.raises(AnalysisError, match="symmetric"):
            oe.simultaneous_ci(
                [0.0, 0.0],
                np.array([[1.0, 0.5], [0.0, 1.0]]) * scale,
                family_description="Materially asymmetric at every scale",
            )
    base = np.array([[4.0, -1.2], [-1.2, 1.0]])
    reference_ci = oe.simultaneous_ci(
        [1.0, 2.0], base, draws=1000, seed=781, family_description="Full covariance"
    )
    for scale in (1e-100, 1e100):
        actual = oe.simultaneous_ci(
            [1.0, 2.0], base * scale**2, draws=1000, seed=781, family_description="Full covariance"
        )
        assert_allclose(
            actual.attrs["critical_value"], reference_ci.attrs["critical_value"], atol=1e-14
        )
        assert_allclose(actual.attrs["joint_covariance"], base * scale**2)
        assert_allclose(actual.std_error, reference_ci.std_error * scale)


@pytest.mark.parametrize("procedure", ["adjust", "stepdown", "ci"])
def test_json_envelope_and_pandas_roundtrip_preserve_table_and_attrs(procedure):
    from io import StringIO
    import pandas as pd

    if procedure == "adjust":
        result = oe.multipletests([0.01, None, 0.1], missing="drop", labels=["a", "b", "c"])
    elif procedure == "stepdown":
        result = oe.stepdown(
            [3.0, 1.0], [[2.0, -1.0], [-2.0, 2.0]], null_description="Known joint law"
        )
    else:
        result = oe.simultaneous_ci(
            [1.0, 2.0],
            [[4.0, -1.0], [-1.0, 9.0]],
            draws=1000,
            family_description="Joint compatible normal covariance",
        )
    state = {
        "schema": 1,
        "table": json.loads(result.to_json(orient="table", double_precision=15)),
        "attrs": result.attrs,
    }
    saved = json.loads(json.dumps(state, allow_nan=False))
    restored = oe.DataFrame(pd.read_json(StringIO(json.dumps(saved["table"])), orient="table"))
    restored.attrs = saved["attrs"]
    if "reject" in restored:
        restored["reject"] = restored["reject"].where(pd.notna(restored["reject"]), None)
    pd.testing.assert_frame_equal(result, restored, check_exact=False, rtol=2e-14, atol=2e-14)
    assert restored.attrs == result.attrs
    assert "\\end{tabular}" in restored.to_latex() or "\\end{longtable}" in restored.to_latex()


def test_unsupported_options_and_unverified_null_design_remain_explicit():
    with pytest.raises(TypeError):
        oe.simultaneous_ci([0.0], [[1.0]], distribution="t", family_description="Unsupported t")
    with pytest.raises(TypeError):
        oe.multipletests([0.1], weights=[1.0])
    with pytest.raises(TypeError):
        oe.stepdown(
            [1.0], [[0.0], [2.0]], resample_rows=True, null_description="Unsupported generation"
        )
    result = oe.stepdown([1.0], [[0.0], [2.0]], null_description="Caller attestation, not proof")
    assert result.attrs["null_design_verified"] is False
    assert result.attrs["dataset_support"] is False


def test_complete_sign_orbit_controls_full_and_partial_null_exactly():
    from itertools import product

    design = np.array(
        [
            [1.0, 0.3, -0.2],
            [0.4, 1.0, 0.5],
            [-0.3, 0.1, 1.0],
            [0.2, -0.4, 0.1],
            [0.5, 0.2, -0.3],
            [-0.2, 0.5, 0.4],
        ]
    )
    design /= np.sqrt((design * design).sum(0))
    orbit = np.array(list(product((-1.0, 1.0), repeat=6))) @ design
    null_p = (np.abs(orbit[:, None, :]) >= np.abs(orbit[None, :, :])).mean(0)
    for partial in (False, True):
        shifted = orbit + ([4.0, 0.0, 0.0] if partial else [0.0, 0.0, 0.0])
        observed_p = (np.abs(orbit[:, None, :]) >= np.abs(shifted[None, :, :])).mean(0)
        for method in ("romano_wolf", "westfall_young"):
            adjusted = []
            for index in range(len(orbit)):
                result = oe.stepdown(
                    observed_p[index] if method == "westfall_young" else shifted[index],
                    null_p if method == "westfall_young" else orbit,
                    method=method,
                    calibration="enumerated",
                    null_description="Complete 64 equiprobable known-sign assignments; subset pivotal location shift",
                )
                adjusted.append(result.adjusted_p_value.to_numpy())
            adjusted = np.array(adjusted)
            for alpha in (0.01, 0.05, 0.1):
                assert np.any(adjusted[:, 1 if partial else 0 :] <= alpha, axis=1).mean() <= alpha


def test_cpu_resident_tensor_contract_and_matrix_dataframe():
    actual = oe.multipletests(torch.tensor([0.01, 0.2], dtype=torch.float64))
    assert actual.adjusted_p_value.tolist() == [0.02, 0.2]
    result = oe.simultaneous_ci(
        [0.0, 0.0],
        pd.DataFrame([[1.0, 0.2], [0.2, 1.0]]),
        draws=1000,
        family_description="Pandas numeric full covariance",
    )
    assert result.attrs["joint_covariance"] == [[1.0, 0.2], [0.2, 1.0]]
    with pytest.raises(AnalysisError, match="CPU"):
        oe.multipletests(torch.empty(2, device="meta"))


def test_explicit_label_covariance_permutations_have_identical_seeded_critical():
    covariance = np.array([[4.0, 0.4, -0.2], [0.4, 1.0, 0.3], [-0.2, 0.3, 9.0]])
    labels, estimates = np.array(["beta", "alpha", "gamma"]), np.array([1.0, 3.0, -2.0])
    original = oe.simultaneous_ci(
        estimates,
        covariance,
        labels=labels,
        seed=6157,
        draws=1000,
        family_description="Identical explicitly identified covariance family",
    )
    assert original.attrs["simulation_order_positions"] == [1, 0, 2]
    for perm in ([2, 0, 1], [1, 2, 0]):
        result = oe.simultaneous_ci(
            estimates[perm],
            covariance[np.ix_(perm, perm)],
            labels=labels[perm],
            seed=6157,
            draws=1000,
            family_description="Identical explicitly identified covariance family",
        )
        assert result.attrs["critical_value"] == original.attrs["critical_value"]
        assert_allclose(result.ci_low, original.ci_low.iloc[perm], atol=0.0, rtol=0.0)
        assert result.attrs["simulation_hypotheses"] == ["alpha", "beta", "gamma"]


def test_runtime_guard_and_torch_free_auxiliary_discovery():
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src") + os.pathsep + str(root / "packages/openecon-charts/src"),
    }
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'scipy','statsmodels','sklearn','linearmodels'}:
   raise ImportError('blocked runtime dependency: '+fullname)
sys.meta_path.insert(0,Block())
from openecon.econometrics.registry import public_exports
entries=public_exports()
assert all(name in entries for name in ('multipletests','stepdown','simultaneous_ci','export_group_state'))
assert 'torch' not in sys.modules
import openecon as oe
assert oe.multipletests([.01,.2]).adjusted_p_value.tolist()==[.02,.2]
assert len(oe.stepdown([1.],[[0.],[2.]],null_description='Known summary null'))==1
assert len(oe.simultaneous_ci([0.],[[1.]],draws=1000,family_description='Known normal covariance'))==1
"""
    process = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True
    )
    assert process.returncode == 0, process.stderr


def test_scientific_runner_exports_numpy_gate_flags_and_keeps_failed_denominator(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/validate_multiple_testing.py"
    spec = importlib.util.spec_from_file_location("multiple_validation_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "FAMILIES", 3)
    monkeypatch.setattr(module, "METHODS", ("holm",))
    monkeypatch.setattr(module, "DESIGNS", ("independent",))
    monkeypatch.setattr(module, "ALPHAS", (0.05,))
    plan = module.protocol()
    native = oe.multipletests
    calls = []

    def one_failure_per_cell(*args, **kwargs):
        index = len(calls)
        calls.append(index)
        if index % 3 == 0:
            raise AnalysisError("isolated_failure", "Retain this planned family's failure.")
        return native(*args, **kwargs)

    monkeypatch.setattr(oe, "multipletests", one_failure_per_cell)
    records = module.adjustments(plan)
    json.dumps(records, allow_nan=False)  # Regresses the observed NumPy-boolean export failure.
    assert len(calls) == 6 and len(records) == 2
    for record in records:
        assert record["passed"] is False
        assert len(record["failures"]) == 1
        assert record["summary"][0]["planned_families"] == 3
        assert record["summary"][0]["failed_families"] == 1
        assert len(record["family_rejection_counts"]["0.05"]["false"]) == 3


@pytest.mark.parametrize("procedure", ["p_values", "covariance", "null_draws"])
def test_float64_conversion_overflow_is_a_controlled_analysis_error(procedure):
    huge = 10**500
    with pytest.raises(AnalysisError, match="numeric|real") as error:
        if procedure == "p_values":
            oe.multipletests([huge])
        elif procedure == "covariance":
            oe.simultaneous_ci(
                [0.0],
                [[huge]],
                draws=1000,
                family_description="Integer cannot be represented in float64 covariance",
            )
        else:
            oe.stepdown([0.0], [[huge], [0]], null_description="Unrepresentable integer null draw")
    assert error.value.code == "invalid_values"
