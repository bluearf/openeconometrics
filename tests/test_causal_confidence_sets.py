"""Independent full-support, finite-grid coverage and exact arithmetic oracles."""

from copy import deepcopy
from fractions import Fraction
from itertools import combinations, product
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design import confidence_sets as module
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.resources import use_workspace_budget


FUNCTIONS = {
    "complete": module.randomization_confidence_set,
    "pair": module.paired_randomization_confidence_set,
    "cluster": module.cluster_randomization_confidence_set,
}
DESIGNS = {"complete": "complete_randomized", "pair": "paired_randomized", "cluster": "cluster_randomized"}


def example(kind):
    if kind == "cluster":
        return pd.DataFrame({"y": [1.25, -2.5, 7., 4.25, 8.5, -1., 9.75],
                             "d": [0, 0, 1, 0, 0, 0, 1],
                             "g": ["a", "a", "b", "c", "c", "c", "d"]},
                            index=[3, 3, "duplicate", "duplicate", 2**60+1, 9, 9])
    return pd.DataFrame({"y": [1.25, -2.5, 7., 4.25, 8.5, -1.],
                         "d": [0, 1, 1, 0, 0, 1], "g": ["a", "a", "b", "b", "c", "c"]},
                        index=[3, 3, "duplicate", "duplicate", 2**60+1, 9])


def fit(kind, data=None, **options):
    data = example(kind) if data is None else data
    extra = [] if kind == "complete" else ["g"]
    return FUNCTIONS[kind](data, "y", "d", *extra,
        candidates=options.pop("candidates", [2., -1., .75]),
        design=options.pop("design", DESIGNS[kind]), **options)


def oracle(data, kind, candidates):
    y = [Fraction(float(value)) for value in data.y]
    d = [int(value) for value in data.d]
    if kind == "complete":
        groups = [[i] for i in range(len(y))]
        unit_bits = [tuple(int(i in chosen) for i in range(len(y)))
                     for chosen in combinations(range(len(y)), sum(d))]
        design_bits = unit_bits
    else:
        groups = []
        for label in dict.fromkeys(data.g):
            positions = [i for i, value in enumerate(data.g) if value == label]
            if kind == "pair":
                positions.sort(key=lambda i: -d[i])
            groups.append(positions)
        if kind == "pair":
            design_bits = list(product((0, 1), repeat=len(groups)))
        else:
            treated = sum(d[rows[0]] for rows in groups)
            design_bits = [tuple(int(i in chosen) for i in range(len(groups)))
                           for chosen in combinations(range(len(groups)), treated)]
        unit_bits = []
        for code in design_bits:
            assignment = [0]*len(y)
            for bit, rows in zip(code, groups, strict=True):
                for i in rows:
                    assignment[i] = bit
                if kind == "pair":
                    assignment[rows[1]] = 1-bit
            unit_bits.append(tuple(assignment))

    def statistic(values, assignment):
        if kind == "pair":
            return sum((value if received else -value for value, received in zip(values, assignment, strict=True)), Fraction())/len(groups)
        if kind == "complete":
            nt = sum(d)
            return sum((values[i] for i, a in enumerate(assignment) if a), Fraction())/nt - sum((values[i] for i, a in enumerate(assignment) if not a), Fraction())/(len(y)-nt)
        nt = sum(d[rows[0]] for rows in groups)
        totals = [sum((values[i] for i in rows), Fraction()) for rows in groups]
        return Fraction(len(groups), len(y))*(sum((totals[g] for g, rows in enumerate(groups) if assignment[rows[0]]), Fraction())/nt
            - sum((totals[g] for g, rows in enumerate(groups) if not assignment[rows[0]]), Fraction())/(len(groups)-nt))

    tests = []
    for effect in candidates:
        adjusted = [value-Fraction(float(effect))*received for value, received in zip(y, d, strict=True)]
        observed = statistic(adjusted, d)
        statistics = [statistic(adjusted, assignment) for assignment in unit_bits]
        flags = [abs(value) >= abs(observed) for value in statistics]
        tests.append(dict(observed=observed, statistics=statistics, extreme=flags,
                          p=Fraction(sum(flags), len(flags))))
    return design_bits, unit_bits, tests


@pytest.mark.parametrize("kind", FUNCTIONS)
@pytest.mark.parametrize("level", [.95, .6, .25])
def test_every_candidate_assignment_cell_fraction_oracle_and_original_state(kind, level):
    data = example(kind)
    original = data.copy(deep=True)
    candidates = [2., -1., .75]
    result = fit(kind, data, candidates=candidates, level=level)
    state = result.attrs["state"]
    bits, units, expected = oracle(data, kind, candidates)
    assert state["assignment_bits"] == ["".join(map(str, code)) for code in bits]
    assert state["unit_assignment_bits"] == ["".join(map(str, code)) for code in units]
    assert state["assignment_universe_size"] == len(bits)
    assert state["assignment_probability_fraction"] == [1, len(bits)]
    assert math.fsum(state["assignment_probabilities"]) == pytest.approx(1)
    for i, reference in enumerate(expected):
        actual = state["candidate_tests"][i]
        assert actual["p_value_fraction"] == [reference["p"].numerator*(len(bits)//reference["p"].denominator), len(bits)]
        assert actual["p_value"] == float(reference["p"])
        assert actual["extreme"] == reference["extreme"]
        assert actual["exact_ties"] == [abs(value) == abs(reference["observed"]) for value in reference["statistics"]]
        assert actual["comparison_signs"] == [int(abs(value) > abs(reference["observed"]))-int(abs(value) < abs(reference["observed"])) for value in reference["statistics"]]
        np.testing.assert_allclose(actual["assignment_statistics"], list(map(float, reference["statistics"])), rtol=3e-15, atol=0)
        assert actual["observed_statistic"] == pytest.approx(float(reference["observed"]), rel=3e-15, abs=0)
        assert actual["accepted"] == (reference["p"] > 1-Fraction(level))
        # Every recorded expansion is an independently replayable exact sum.
        power = Fraction(2)**state["normalization_power_of_two"]
        multiplier = Fraction(1, 1)
        if kind == "complete":
            multiplier = Fraction(1, sum(data.d)*(len(data)-sum(data.d)))
        elif kind == "pair":
            multiplier = Fraction(1, data.g.nunique())
        else:
            g, nt = data.g.nunique(), data.groupby("g").d.first().sum()
            multiplier = Fraction(int(g), int(len(data)*nt*(g-nt)))
        for column, truth in enumerate([reference["observed"], *reference["statistics"]]):
            replay = sum((Fraction(part[column]) for part in actual["normalized_numerator_expansion"]), Fraction())
            assert replay*multiplier/power == truth
    assert result["profile"].effect.tolist() == candidates
    assert result.attrs["positions"] == list(range(len(data)))
    assert result.attrs["unit_labels"] == list(data.index)
    assert len(result["tests"]) == len(bits)*len(candidates)
    assert result.attrs["state"]["confidence_interval"] is None
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_exact_true_candidate_design_coverage_by_full_randomized_data_enumeration(kind):
    if kind == "cluster":
        data = pd.DataFrame({"y": [0., 3., 1., -2., 4., 6.], "d": [1, 1, 0, 1, 1, 0],
                             "g": ["a", "a", "b", "c", "c", "d"]})
    else:
        data = example(kind).reset_index(drop=True)
        data.y = [0., 3., 1., -2., 4., 6.]
    y0 = data.y.to_numpy().copy()
    _, assignments, _ = oracle(data, kind, [0])
    theta = 2.
    level = .75
    included = 0
    for assignment in assignments:
        realized = data.copy()
        realized.d = assignment
        realized.y = y0+theta*np.asarray(assignment)
        result = fit(kind, realized, candidates=[-1., theta, 5.], level=level)
        _, _, expected = oracle(realized, kind, [-1., theta, 5.])
        assert result["profile"].accepted.tolist() == [entry["p"] > 1-Fraction(level) for entry in expected]
        included += int(result["profile"].accepted.iloc[1])
    assert Fraction(included, len(assignments)) >= Fraction(level)


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_severe_nested_cancellation_and_math_fsum_reference(kind):
    if kind == "complete":
        data = pd.DataFrame({"y": [1e16, 1., -1e16, 0., 0., 0.], "d": [1, 1, 1, 0, 0, 0]})
    elif kind == "pair":
        data = pd.DataFrame({"y": [1e16, 0., 1., 0., -1e16, 0.], "d": [1, 0, 1, 0, 1, 0], "g": [0, 0, 1, 1, 2, 2]})
    else:
        data = pd.DataFrame({"y": [1e16, 1., -1e16, 0., 0., 0.], "d": [1, 1, 1, 1, 0, 0], "g": [0, 0, 1, 1, 2, 3]})
    result = fit(kind, data, candidates=[0])
    _, _, expected = oracle(data, kind, [0])
    assert expected[0]["observed"] == Fraction(1, 3)
    assert result["profile"].observed_statistic.iloc[0] == float(Fraction(1, 3))
    assert result.attrs["state"]["candidate_tests"][0]["extreme"] == expected[0]["extreme"]
    assert math.fsum(data.y[data.d == 1]) == 1


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_nondyadic_null_low_parts_preserve_every_exact_comparison(kind):
    data = example(kind)
    data.y = [1.1, -2.3, 7.2, 4.1, 8.3, -1.4] + ([9.2] if kind == "cluster" else [])
    candidates = [.3, 1.1]
    result = fit(kind, data, candidates=candidates)
    _, _, expected = oracle(data, kind, candidates)
    assert any(value != 0 for record in result.attrs["state"]["candidate_tests"] for value in record["null_imputation_low_parts"])
    for actual, reference in zip(result.attrs["state"]["candidate_tests"], expected, strict=True):
        assert actual["extreme"] == reference["extreme"]
        assert actual["p_value"] == float(reference["p"])


def test_actual_disconnected_empty_and_all_candidate_sets_have_no_hull():
    data = pd.DataFrame({"y": [-3.]*7, "d": [1, 1, 1, 1, 1, 0, 0],
                         "g": ["a", "b", "b", "b", "c", "d", "d"]})
    grid = list(range(-12, 13))
    result = fit("cluster", data, candidates=grid, level=.75)
    _, _, expected = oracle(data, "cluster", grid)
    keep = [entry["p"] > Fraction(1, 4) for entry in expected]
    assert result["profile"].accepted.tolist() == keep
    assert result["accepted_candidates"].effect.tolist() == [-12., -11., -10., -9., *map(float, range(-1, 13))]
    assert result.attrs["state"]["disconnected_on_sorted_grid"]
    assert len(result.attrs["state"]["accepted_grid_components"]) == 2
    assert result.attrs["state"]["confidence_interval"] is None
    empty = fit("cluster", data, candidates=[-8, -7], level=.75)
    assert empty["accepted_candidates"].empty
    assert empty.attrs["state"]["empty_set"]
    pd.testing.assert_frame_equal(causal_design_load(causal_design_save(empty))["accepted_candidates"], empty["accepted_candidates"], check_exact=True)
    full = fit("cluster", data, candidates=[0, 1], level=.75)
    assert full.attrs["state"]["all_candidates_accepted"]
    assert full.attrs["state"]["parameter_space"] == "supplied finite candidate grid only"


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_nonrepresentable_original_unit_statistic_refused(kind):
    data = example(kind)
    data.y = [math.ulp(0.)]+[0.]*(len(data)-1)
    with pytest.raises(AnalysisError, match="underflows"):
        fit(kind, data, candidates=[0])


def test_flush_denormal_capability_refusal_does_not_mutate_caller_mode():
    # Isolate the intentionally changed floating-point environment from pytest
    # and from the user's analysis process; production never calls the setter.
    code = """
import pandas as pd
import torch
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design.confidence_sets import randomization_confidence_set
assert torch.set_flush_denormal(True)
tiny=torch.tensor(torch.finfo(torch.float64).tiny,dtype=torch.float64)
assert bool(tiny*.5==0)
try:
    randomization_confidence_set(pd.DataFrame({'y':[1.,2.],'d':[0,1]}),'y','d',candidates=[0],design='complete_randomized')
except AnalysisError as error:
    assert error.code=='numerical_failure' and 'FTZ/DAZ' in str(error)
else:
    raise AssertionError('flush-denormal arithmetic was accepted')
assert bool(tiny*.5==0)
"""
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True, capture_output=True, text=True)


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_large_common_location_exact_ties_and_power_of_two_rescaling(kind):
    data = example(kind)
    data.y = 1e150
    result = fit(kind, data, candidates=[0])
    _, _, expected = oracle(data, kind, [0])
    assert result.attrs["state"]["candidate_tests"][0]["extreme"] == expected[0]["extreme"]
    assert result.attrs["state"]["candidate_tests"][0]["exact_ties"] == [abs(v) == abs(expected[0]["observed"]) for v in expected[0]["statistics"]]
    if kind != "cluster":
        assert result["profile"].observed_statistic.iloc[0] == 0


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_subnormal_supported_rescaling_preserves_p_values(kind):
    data = example(kind)
    # Dyadic inputs, far above the smallest subnormal after report denominators.
    data.y = np.ldexp(data.y.to_numpy(), -1040)
    candidates = list(np.ldexp(np.asarray([2., -1., .75]), -1040))
    result = fit(kind, data, candidates=candidates)
    _, _, expected = oracle(data, kind, candidates)
    assert result["profile"].p_value.tolist() == [float(x["p"]) for x in expected]
    for actual, reference in zip(result.attrs["state"]["candidate_tests"], expected, strict=True):
        assert actual["extreme"] == reference["extreme"]


@pytest.mark.parametrize("kind", FUNCTIONS)
@pytest.mark.parametrize("candidates", [[], list(range(65)), [0, 0], [False], [True], [math.nan], [math.inf], ["0"], [2**60+1], "0"])
def test_strict_fixed_grid_roles(kind, candidates):
    with pytest.raises(AnalysisError):
        fit(kind, candidates=candidates)


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_exact_numeric_candidate_admission_preserves_original_fraction_and_native_float(kind):
    with pytest.raises(AnalysisError, match="represented exactly"):
        fit(kind, candidates=[Fraction(1, 3)])
    assert fit(kind, candidates=[Fraction(1, 2)])["profile"].effect.tolist() == [.5]
    native = np.longdouble(1) + np.ldexp(np.longdouble(1), -60)
    if native != float(native):
        with pytest.raises(AnalysisError, match="represented exactly"):
            fit(kind, candidates=[native])
    else:
        assert fit(kind, candidates=[native])["profile"].effect.tolist() == [float(native)]


@pytest.mark.parametrize("kind", FUNCTIONS)
@pytest.mark.parametrize("key,value", [("missing", "drop"), ("device", "cuda"), ("weights", "w"),
                                       ("design", "randomized"), ("level", 0), ("level", 1), ("level", True)])
def test_unsupported_options_fail(kind, key, value):
    with pytest.raises(AnalysisError):
        fit(kind, **{key: value})


@pytest.mark.parametrize("kind", FUNCTIONS)
@pytest.mark.parametrize("role", ["y", "d", "index", "label"])
def test_missing_and_bounded_escaped_original_labels(kind, role):
    data = example(kind)
    if role in ("y", "d"):
        data.loc[data.index[0], role] = math.nan
    elif role == "index":
        data.index = ["\x00"*43]+list(data.index[1:])
    elif kind != "complete":
        data.g = "\x00"*43
    else:
        data.index = ["x"*257]+list(data.index[1:])
    with pytest.raises(AnalysisError):
        fit(kind, data)


@pytest.mark.parametrize("kind", ["pair", "cluster"])
def test_broken_original_topology_refused_before_nulls(kind):
    data = example(kind)
    data.iloc[0, data.columns.get_loc("d")] = 1
    with pytest.raises(AnalysisError):
        fit(kind, data)


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_all_candidate_nulls_preflight_before_assignments(kind, monkeypatch):
    data = example(kind)
    data.y = 1e150
    calls = []
    monkeypatch.setattr(module, "_assignments", lambda *a, **k: calls.append(True))
    with pytest.raises(AnalysisError, match="absorbed"):
        fit(kind, data, candidates=[0, 1])
    assert calls == []


@pytest.mark.parametrize("kind", FUNCTIONS)
@pytest.mark.parametrize("dtype", ["int64", "Int64", "uint64"])
def test_original_integer_outcomes_refused_before_numeric_or_support_allocation(kind, dtype, monkeypatch):
    data = example(kind)
    data["y"] = pd.array([2**53+1]*len(data), dtype=dtype)
    assert int(data.y.iloc[0]) == 2**53+1
    assert int(float(data.y.iloc[0])) != int(data.y.iloc[0])
    monkeypatch.setattr(module.m, "tensor", lambda *a, **k: pytest.fail("Numeric allocation preceded original integer admission."))
    monkeypatch.setattr(module, "_assignments", lambda *a, **k: pytest.fail("Support allocation preceded original integer admission."))
    with pytest.raises(AnalysisError, match="integer outcomes.*exactly"):
        fit(kind, data, candidates=[0])


@pytest.mark.parametrize("kind", FUNCTIONS)
@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_boolean_outcomes_refused_before_support_allocation(kind, dtype, monkeypatch):
    data = example(kind)
    data["y"] = pd.array([True]*len(data), dtype=dtype)
    monkeypatch.setattr(module, "_assignments", lambda *a, **k: pytest.fail("Support allocation preceded outcome type admission."))
    with pytest.raises(AnalysisError, match="Boolean"):
        fit(kind, data, candidates=[0])


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_native_extended_float_outcome_admission_before_numeric_and_support_allocation(kind, monkeypatch):
    data = example(kind)
    data["y"] = np.array(data.y, dtype=np.longdouble)
    if data.y.dtype.itemsize > 8:
        monkeypatch.setattr(module.m, "tensor", lambda *a, **k: pytest.fail("Numeric allocation preceded extended-float admission."))
        monkeypatch.setattr(module, "_assignments", lambda *a, **k: pytest.fail("Support allocation preceded extended-float admission."))
        with pytest.raises(AnalysisError, match="wider than float64"):
            fit(kind, data, candidates=[0])
    else:
        assert fit(kind, data, candidates=[0])["profile"].effect.tolist() == [0.]


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_exact_large_integer_outcomes_are_admitted_without_arbitrary_magnitude_cap(kind):
    data = example(kind)
    data["y"] = pd.array([2**60]*len(data), dtype="int64")
    result = fit(kind, data, candidates=[0])
    _, _, reference = oracle(data, kind, [0])
    assert result.attrs["state"]["candidate_tests"][0]["extreme"] == reference[0]["extreme"]
    assert result.attrs["state"]["outcomes"] == [float(2**60)]*len(data)


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_multiplicative_work_and_workspace_preflight_no_partial_support(kind, monkeypatch):
    calls = []
    monkeypatch.setattr(module, "_assignments", lambda *a, **k: calls.append(True))
    with pytest.raises(AnalysisError, match="max_work"):
        fit(kind, candidates=list(range(64)), max_work=1000)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        fit(kind, candidates=list(range(64)), max_work=1_000_000_000)
    assert calls == []


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_complete_artifact_roundtrip_tamper_latex_and_private_cpu_state(kind):
    torch.manual_seed(519)
    before_rng = torch.get_rng_state().clone()
    result = fit(kind)
    artifact = causal_design_save(result)
    assert torch.equal(before_rng, torch.get_rng_state())

    restored = causal_design_load(artifact)
    assert causal_design_save(restored) == artifact
    for name in result:
        pd.testing.assert_frame_equal(restored[name], result[name], check_exact=True)
    assert restored.attrs == result.attrs
    assert "\\begin{tabular}" in result.to_latex()
    damaged = deepcopy(artifact)
    damaged["payload"]["attrs"]["state"]["candidate_tests"][0]["extreme"][0] = not damaged["payload"]["attrs"]["state"]["candidate_tests"][0]["extreme"][0]
    with pytest.raises(AnalysisError, match="artifact"):
        causal_design_load(damaged)
    original = torch.get_default_dtype()
    try:
        for dtype in [torch.float32, torch.float64]:
            torch.set_default_dtype(dtype)
            with torch.device("meta"):
                other = causal_design_save(fit(kind))
                assert torch.empty(0).device.type == "meta"
            assert other == artifact
    finally:
        torch.set_default_dtype(original)
    assert torch.equal(before_rng, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["pair", "cluster"])
def test_typed_identities_do_not_conflate_bool_integer_float_or_large_index(kind):
    big = 2**60+1
    identities = pd.Series([True, True, 1, 1, big, big, 1.5, 1.5], dtype=object)
    d = [0, 1, 1, 0, 0, 1, 1, 0] if kind == "pair" else [0, 0, 1, 1, 0, 0, 1, 1]
    data = pd.DataFrame({"y": list(map(float, range(8))), "d": d, "g": identities})
    result = fit(kind, data, candidates=[0])
    assert result["design"].label_type.tolist() == ["bool", "int", "int", "float"]
    assert result["design"].label.iloc[2] == big
    assert result.attrs["state"]["groups"][2]["label"] == big
    assert causal_design_save(causal_design_load(causal_design_save(result))) == causal_design_save(result)
