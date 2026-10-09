"""Independent transport LP, complete-coupling and exact-rational oracles."""

import copy
from fractions import Fraction
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.causal_design import common as m
from openecon.econometrics.causal_design import effect_distribution as e
from openecon.resources import use_workspace_budget


FIXTURES = [
    ([0, 1], [1, 1], [1, 1]),
    ([-2, 0, 3], [1, 1, 1], [1, 1, 1]),
    ([-1, 0, 2], [2, 1, 0], [0, 1, 2]),
]


def data_for(support, c0, c1):
    return pd.DataFrame(
        dict(
            y=[
                value
                for counts in (c0, c1)
                for value, count in zip(support, counts)
                for _ in range(count)
            ],
            d=[0] * sum(c0) + [1] * sum(c1),
        )
    )


def cdf(data=None, **options):
    return e.treatment_effect_cdf_bounds(
        data_for(*FIXTURES[0]) if data is None else data,
        "y",
        "d",
        **(dict(support=[0, 1], thresholds=[-1, 0, 1], design="randomized") | options),
    )


def quantile(data=None, **options):
    return e.treatment_effect_quantile_bounds(
        data_for(*FIXTURES[0]) if data is None else data,
        "y",
        "d",
        **(dict(support=[0, 1], quantiles=[0.25, 0.5, 0.75], design="randomized") | options),
    )


def enumerate_couplings(rows, columns):
    """All integer transportation tables, independent Python composition recursion."""
    size = len(rows)

    def recurse(i, remaining, prefix):
        if i == size - 1:
            if sum(remaining) == rows[i]:
                yield np.array(prefix + [remaining], dtype=np.int64)
            return
        for left in itertools.product(*(range(min(rows[i], x) + 1) for x in remaining[:-1])):
            final = rows[i] - sum(left)
            if 0 <= final <= remaining[-1]:
                row = [*left, final]
                yield from recurse(i + 1, [a - b for a, b in zip(remaining, row)], prefix + [row])

    yield from recurse(0, list(columns), [])


def transport_lp(c0, c1, support, threshold):
    size = len(support)
    matrix = np.zeros((2 * size, size * size))
    for i in range(size):
        for j in range(size):
            matrix[i, i * size + j] = 1
            matrix[size + j, i * size + j] = 1
    margins = np.array([*c0, *c1], dtype=float)
    margins[:size] /= sum(c0)
    margins[size:] /= sum(c1)
    objective = np.array([int(b - a <= threshold) for a in support for b in support])
    lower = linprog(objective, A_eq=matrix, b_eq=margins, bounds=(0, None), method="highs")
    upper = linprog(-objective, A_eq=matrix, b_eq=margins, bounds=(0, None), method="highs")
    assert lower.success and upper.success
    return [lower.fun, -upper.fun]


def assert_certificate(certificate, row_margins, column_margins, mask):
    """Check primal+dual optimality from saved arrays, without calling the solver."""
    size = len(row_margins)
    capacities = np.array(certificate["capacities"], dtype=np.int64)
    flow = np.array(certificate["flow"], dtype=np.int64)
    residual = np.array(certificate["residual"], dtype=np.int64)
    coupling = np.array(certificate["coupling_mass_counts"], dtype=np.int64)
    cut = np.array(certificate["reachable_cut"], dtype=bool)
    mass = sum(row_margins)
    expected = np.zeros((2 * size + 2, 2 * size + 2), dtype=np.int64)
    expected[2 * size, :size] = row_margins
    expected[size : 2 * size, 2 * size + 1] = column_margins
    expected[:size, size : 2 * size] = mask * mass
    np.testing.assert_array_equal(capacities, expected)
    np.testing.assert_array_equal(flow, -flow.T)
    np.testing.assert_array_equal(residual, capacities - flow)
    assert (residual >= 0).all()
    balances = np.zeros(2 * size + 2, dtype=np.int64)
    balances[2 * size] = certificate["maximum_allowed_mass_count"]
    balances[-1] = -balances[-2]
    np.testing.assert_array_equal(flow.sum(1), balances)
    assert cut[-2] and not cut[-1] and not (residual[cut][:, ~cut] > 0).any()
    reachable = {2 * size}
    queue = [2 * size]
    for node in queue:
        for other in np.where(residual[node] > 0)[0]:
            if int(other) not in reachable:
                reachable.add(int(other))
                queue.append(int(other))
    assert set(np.where(cut)[0]) == reachable
    assert capacities[cut][:, ~cut].sum() == certificate["cut_capacity"] == balances[-2]
    assert (coupling >= 0).all()
    np.testing.assert_array_equal(coupling.sum(1), row_margins)
    np.testing.assert_array_equal(coupling.sum(0), column_margins)
    assert coupling[mask].sum() == balances[-2]


@pytest.mark.parametrize("support,c0,c1", FIXTURES)
def test_cdf_all_support_boundaries_full_enumeration_and_independent_continuous_lp(support, c0, c1):
    differences = np.array([[b - a for b in support] for a in support])
    grid = sorted(set(differences.flatten()))
    thresholds = sorted(
        {
            -100.0,
            100.0,
            *(math.nextafter(x, -math.inf) for x in grid),
            *grid,
            *(math.nextafter(x, math.inf) for x in grid),
        },
        reverse=True,
    )
    result = cdf(data_for(support, c0, c1), support=support, thresholds=thresholds)
    state = result.attrs["state"]
    mass = sum(c0) * sum(c1)
    rows = np.array(c0) * sum(c1)
    columns = np.array(c1) * sum(c0)
    couplings = list(enumerate_couplings(rows, columns))
    assert couplings
    assert list(result["bounds"].threshold) == thresholds
    for threshold, record in zip(thresholds, state["extrema"]):
        mask = differences <= threshold
        counts = [int(coupling[mask].sum()) for coupling in couplings]
        assert [record["lower_mass_count"], record["upper_mass_count"]] == [
            min(counts),
            max(counts),
        ]
        np.testing.assert_allclose(
            np.array([min(counts), max(counts)]) / mass,
            transport_lp(c0, c1, support, threshold),
            atol=1e-15,
        )
        for bound, objective in [("lower", min(counts)), ("upper", max(counts))]:
            certificate = record[bound]
            assert_certificate(certificate, rows, columns, mask if bound == "upper" else ~mask)
            witness = np.array(certificate["coupling_mass_counts"])
            assert int(witness[mask].sum()) == objective
    assert state["inference"] is None and state["covariance"] is None
    assert state["standard_error"] is None and state["confidence_interval"] is None
    assert not state["joint_curve_sharpness_claimed"] and not state["population_sampling_inference"]


@pytest.mark.parametrize("support,c0,c1", FIXTURES)
def test_quantile_full_enumeration_exact_fraction_crossings_and_attaining_witness(support, c0, c1):
    mass = sum(c0) * sum(c1)
    fractions = {1 / mass, 1 / 3, 0.5, 0.75}
    qs = sorted(
        {
            math.nextafter(0.0, 1.0),
            math.nextafter(1.0, 0.0),
            *(math.nextafter(q, -math.inf) for q in fractions),
            *fractions,
            *(math.nextafter(q, math.inf) for q in fractions),
        }
    )
    result = quantile(data_for(support, c0, c1), support=support, quantiles=qs)
    state = result.attrs["state"]
    differences = np.array([[b - a for b in support] for a in support])
    grid = sorted(set(differences.flatten()))
    couplings = list(enumerate_couplings(np.array(c0) * sum(c1), np.array(c1) * sum(c0)))
    for q, record in zip(qs, state["quantile_records"]):
        exact = Fraction.from_float(q) * mass
        required = (exact.numerator + exact.denominator - 1) // exact.denominator
        assert required == record["required_mass_count"]
        assert Fraction(
            int(record["quantile_numerator"]), int(record["quantile_denominator"])
        ) == Fraction.from_float(q)
        values = [
            next(x for x in grid if int(coupling[differences <= x].sum()) >= required)
            for coupling in couplings
        ]
        assert [record["lower_bound"], record["upper_bound"]] == [min(values), max(values)]
        for bound in ("lower", "upper"):
            witness = record[bound + "_witness"]
            coupling = np.array(witness["coupling_mass_counts"])
            pmf = [int(coupling[differences == x].sum()) for x in grid]
            assert witness["effect_mass_counts"] == pmf
            assert witness["cdf_mass_counts"] == np.cumsum(pmf).tolist()
            assert (
                next(x for x, value in zip(grid, np.cumsum(pmf)) if value >= required)
                == record[bound + "_bound"]
            )
    assert list(result["bounds"]["quantile"]) == qs


def test_upper_quantile_uses_predecessor_witness_instead_of_current_threshold_minimizer():
    state = quantile(quantiles=[0.5]).attrs["state"]
    record = state["quantile_records"][0]
    assert record["lower_bound"] == -1 and record["upper_bound"] == 0
    assert record["upper_certificate_predecessor_index"] == 0
    # At the current threshold0, a valid CDF minimizer gives Q(.5)=-1.
    current = state["extrema"][1]["lower"]["coupling_mass_counts"]
    assert current == [[0, 2], [2, 0]]
    assert record["upper_witness"]["coupling_mass_counts"] == [[2, 0], [0, 2]]
    assert state["arm_counts"] == [[1, 1], [1, 1]]  # Both marginal QTEs are0.


def test_singleton_support_and_first_grid_upper_quantile_complete_witness():
    data = pd.DataFrame(dict(y=[0.125] * 5, d=[0, 1, 1, 1, 1]))
    a = cdf(
        data,
        support=[0.125],
        thresholds=[math.nextafter(0, -math.inf), 0, math.nextafter(0, math.inf)],
    )
    assert list(a["bounds"].lower_bound) == [0, 1, 1]
    assert list(a["bounds"].upper_bound) == [0, 1, 1]
    b = quantile(data, support=[0.125], quantiles=[math.nextafter(0, 1), 0.5, math.nextafter(1, 0)])
    assert list(b["bounds"].lower_bound) == list(b["bounds"].upper_bound) == [0, 0, 0]
    for record in b.attrs["state"]["quantile_records"]:
        assert record["upper_certificate_predecessor_index"] is None
        assert record["upper_witness"]["coupling_mass_counts"] == [[4]]


@pytest.mark.parametrize(
    "support",
    [
        [0, 1],
        [3, -2, 0],
        [0, 0.125, 0.25, 0.375],
        [math.ldexp(1.0, -1074), math.ldexp(2.0, -1074)],
        [1e100, math.nextafter(1e100, math.inf)],
        [-1e100, 1e100],
        [0, 0.1, 0.2, 0.3],
        [1, 1e16],
        [-1e150, 1e150],
        [1e151, 2e151],
    ],
)
def test_native_twosum_admission_matches_independent_exact_fraction_differences(support):
    exact = (
        all(
            Fraction.from_float(b) - Fraction.from_float(a) == Fraction.from_float(b - a)
            for a in map(float, support)
            for b in map(float, support)
        )
        and max(map(abs, support)) <= 1e150
    )
    if exact:
        observed = e._differences(support).tolist()
        assert observed == [[float(b - a) for b in support] for a in support]
    else:
        with pytest.raises(AnalysisError) as caught:
            e._differences(support)
        assert caught.value.code == "numerical_failure"


def test_nonrepresentable_collision_refuses_before_sample_or_flow(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No sample/flow allocation is allowed before support admission")

    monkeypatch.setattr(m, "sample", forbidden)
    monkeypatch.setattr(e, "_flow", forbidden)
    with pytest.raises(AnalysisError, match="exactly representable"):
        cdf(pd.DataFrame(dict(y=[1, 10**16], d=[0, 1])), support=[1, 10**16])


def test_input_integer_collision_is_not_silently_coerced_to_support():
    with pytest.raises(AnalysisError, match="integer outcome"):
        cdf(pd.DataFrame(dict(y=[2**53 + 1, 2**53], d=[0, 1])), support=[float(2**53)])
    with pytest.raises(AnalysisError, match="integer support"):
        cdf(support=[2**53 + 1])


def test_support_order_and_observation_order_remain_original_and_do_not_change_bounds():
    data = data_for(*FIXTURES[1])
    data.index = pd.Index(["duplicate", 1, "1", True, -(2**63), 2**63 - 1], dtype=object)
    a = cdf(data, support=[3, -2, 0], thresholds=[1, -3, 0])
    b = cdf(data.iloc[::-1], support=[0, 3, -2], thresholds=[1, -3, 0])
    pd.testing.assert_frame_equal(a["bounds"], b["bounds"])
    state = a.attrs["state"]
    assert state["support"] == [3.0, -2.0, 0.0]
    assert state["outcomes"] == list(map(float, data.y))
    assert state["sample"]["positions"] == list(range(6))
    assert state["sample"]["unit_labels"] == [m._label(x) for x in data.index]
    assert state["sample"]["input_schema"] == {"y": "int64", "d": "int64"}
    assert list(a["couplings"].control_support_index[:9]) == [0] * 3 + [1] * 3 + [2] * 3


@pytest.mark.parametrize("fit", [cdf, quantile])
def test_complete_json_typed_roundtrip_determinism_no_refit_and_latex(fit, tmp_path, monkeypatch):
    result = fit()
    artifact = m.causal_design_save(result, tmp_path / "complete.json")
    assert json.loads((tmp_path / "complete.json").read_text()) == artifact

    def forbidden(*args, **kwargs):
        pytest.fail("Artifact loading must not solve transportation problems")

    monkeypatch.setattr(e, "_flow", forbidden)
    restored = m.causal_design_load(tmp_path / "complete.json")
    e.validate_effect_distribution_result(restored)
    assert m.causal_design_save(restored) == artifact
    assert result.attrs == restored.attrs
    assert list(result) == list(restored)
    for name in result:
        pd.testing.assert_frame_equal(result[name], restored[name], check_exact=True)
    assert "tabular" in restored["bounds"].to_latex()


@pytest.mark.parametrize("fit", [cdf, quantile])
def test_no_randomness_global_torch_dtype_or_default_device_changes(fit):
    np_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    device = torch.get_default_device()
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("meta")
    try:
        a = fit()
        b = fit()
        assert m.causal_design_save(a) == m.causal_design_save(b)
        assert torch.get_default_dtype() == torch.float32
        assert str(torch.get_default_device()) == "meta"
    finally:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    after = np.random.get_state()
    assert (
        np_before[0] == after[0]
        and np.array_equal(np_before[1], after[1])
        and np_before[2:] == after[2:]
    )


@pytest.mark.parametrize(
    "options,code",
    [
        ({"device": "cuda"}, "unsupported_device"),
        ({"weights": "w"}, "unsupported_weights"),
        ({"design": "observational"}, "unsupported_design"),
        ({"design": True}, "unsupported_design"),
        ({"missing": "drop"}, "unsupported_missing"),
        ({"support": []}, "invalid_spec"),
        ({"support": list(range(9))}, "invalid_spec"),
        ({"support": [0, 0]}, "invalid_spec"),
        ({"support": [0, math.inf]}, "invalid_option"),
        ({"support": [10**1000]}, "numerical_failure"),
        ({"support": [False, 1]}, "invalid_option"),
        ({"thresholds": [0, 0]}, "invalid_spec"),
        ({"thresholds": list(range(65))}, "invalid_spec"),
        ({"max_work": 1}, "work_budget_exceeded"),
        ({"max_work": True}, "invalid_option"),
    ],
)
def test_explicit_cdf_domain_guards(options, code):
    with pytest.raises(AnalysisError) as caught:
        cdf(**options)
    assert caught.value.code == code


@pytest.mark.parametrize(
    "qs", [[0], [1], [-0.1], [math.nan], [True], [], [0.5, 0.5], [i / 18 for i in range(1, 18)]]
)
def test_quantile_domain_guards(qs):
    with pytest.raises(AnalysisError):
        quantile(quantiles=qs)


@pytest.mark.parametrize(
    "data,code",
    [
        (pd.DataFrame(dict(y=[0, 1], d=[0, 0])), "invalid_treatment"),
        (pd.DataFrame(dict(y=[0, 1], d=[0, 2])), "invalid_treatment"),
        (pd.DataFrame(dict(y=[0, 1], d=[False, True])), "invalid_treatment"),
        (pd.DataFrame(dict(y=[False, True], d=[0, 1])), "non_numeric_column"),
        (pd.DataFrame(dict(y=[0, np.nan], d=[0, 1])), "missing_values"),
        (pd.DataFrame(dict(y=[0, math.inf], d=[0, 1])), "non_finite_values"),
        (pd.DataFrame(dict(y=[0, 2], d=[0, 1])), "invalid_support"),
        (pd.DataFrame(dict(y=["0", "1"], d=[0, 1])), "non_numeric_column"),
        (pd.DataFrame(dict(y=[0], d=[1])), "insufficient_observations"),
    ],
)
def test_full_original_sample_guards(data, code):
    with pytest.raises(AnalysisError) as caught:
        cdf(data)
    assert caught.value.code == code


def test_dataset_and_overlapping_roles_are_explicitly_refused(tmp_path):
    path = tmp_path / "data.csv"
    data_for(*FIXTURES[0]).to_csv(path, index=False)
    with pytest.raises(AnalysisError) as caught:
        cdf(Dataset())
    assert caught.value.code == "unsupported_dataset"
    with pytest.raises(AnalysisError, match="distinct"):
        e.treatment_effect_cdf_bounds(
            data_for(*FIXTURES[0]), "y", "y", support=[0, 1], thresholds=[0], design="randomized"
        )
    with pytest.raises(TypeError):
        e.treatment_effect_cdf_bounds(
            data_for(*FIXTURES[0]), "y", "d", support=[0, 1], thresholds=[0]
        )


def test_full_work_and_workspace_preflight_before_sampling_difference_tensor_or_solver(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Resource rejection must precede numerical allocations and sample copy")

    monkeypatch.setattr(m, "sample", forbidden)
    monkeypatch.setattr(e, "_differences", forbidden)
    monkeypatch.setattr(e, "_flow", forbidden)
    with pytest.raises(AnalysisError) as caught:
        cdf(max_work=1)
    assert caught.value.code == "work_budget_exceeded"
    with use_workspace_budget(1), pytest.raises(AnalysisError) as caught:
        cdf(support=list(range(8)), thresholds=list(range(64)), max_work=1_000_000_000)
    assert caught.value.code == "workspace_limit"
    with pytest.raises(AnalysisError) as caught:
        quantile(support=list(range(8)))
    assert caught.value.code == "work_budget_exceeded"


def test_actual_json_escaped_original_index_label_budget():
    data = data_for(*FIXTURES[0])
    data.index = ["\x00" * 43, "a", "b", "c"]
    with pytest.raises(AnalysisError) as caught:
        cdf(data)
    assert caught.value.code == "resource_limit"
    data.index = ["\x00" * 30, "a", "b", "c"]
    result = cdf(data)
    actual = sum(len(m.canonical(m._label(label)).encode("ascii")) for label in data.index) + sum(
        len(m.canonical(m._label(name)).encode("ascii")) for name in ("y", "d")
    )
    assert result.attrs["state"]["sample"]["original_escaped_label_bytes"] == actual


@pytest.mark.parametrize(
    "path,value",
    [
        (("extrema", 0, "upper", "coupling_mass_counts", 0, 0), 1),
        (("extrema", 0, "upper", "flow", 0, 2), 1),
        (("extrema", 0, "upper", "residual", 0, 2), 1),
        (("extrema", 0, "upper", "cut_capacity"), 99),
        (("extrema", 0, "upper", "reachable_cut", 0), False),
        (("extrema", 0, "upper", "augmentations"), -1),
        (("extrema", 0, "upper_mass_count"), 0),
        (("row_mass_counts", 0), 3),
        (("pairwise_effects", 0, 1), 0),
        (("settings", "support"), [0, 2]),
        (("population_sampling_inference",), True),
        (("confidence_interval",), [0, 1]),
    ],
)
def test_pure_semantic_validator_rejects_rehashed_transport_or_domain_tampering(path, value):
    result = copy.deepcopy(cdf())
    state = result.attrs["state"]
    item = state
    for part in path[:-1]:
        item = item[part]
    item[path[-1]] = value
    result.attrs["state_sha256"] = m.digest(state)
    with pytest.raises(AnalysisError):
        e.validate_effect_distribution_result(result)


@pytest.mark.parametrize(
    "bound,field,value",
    [
        ("upper_witness", "effect_mass_counts", [1, 1, 2]),
        ("upper_witness", "coupling_mass_counts", [[0, 2], [2, 0]]),
        (None, "required_mass_count", 1),
        (None, "upper_certificate_predecessor_index", 1),
    ],
)
def test_pure_semantic_validator_rejects_wrong_quantile_crossings_and_witnesses(
    bound, field, value
):
    result = copy.deepcopy(quantile(quantiles=[0.5]))
    record = result.attrs["state"]["quantile_records"][0]
    (record[bound] if bound else record)[field] = value
    with pytest.raises(AnalysisError) as caught:
        e.validate_effect_distribution_result(result)
    assert caught.value.code == "invalid_saved_result"


def test_full_table_numeric_dtype_and_index_tampering_rejected():
    for change in ("value", "dtype", "index"):
        result = copy.deepcopy(cdf())
        if change == "value":
            result["bounds"].iloc[0, 1] = 0.5
        elif change == "dtype":
            result["bounds"]["lower_mass_count"] = result["bounds"]["lower_mass_count"].astype(
                float
            )
        else:
            result["bounds"].index = [2, 1, 0]
        result.attrs["tables_sha256"] = m.digest(m._tables(result))
        with pytest.raises(AnalysisError) as caught:
            e.validate_effect_distribution_result(result)
        assert caught.value.code == "invalid_saved_result"


@pytest.mark.parametrize("fit", [cdf, quantile])
def test_rehashed_full_artifact_semantic_tamper_is_rejected_on_load_and_save(fit):
    result = fit()
    artifact = m.causal_design_save(result)
    changed = copy.deepcopy(artifact)
    state = changed["payload"]["attrs"]["state"]
    if fit is cdf:
        state["extrema"][0]["upper"]["coupling_mass_counts"][0][0] = 1
    else:
        state["quantile_records"][0]["upper_certificate_predecessor_index"] = 2
    changed["payload"]["attrs"]["state_sha256"] = m.digest(state)
    changed["sha256"] = m.digest(changed["payload"])
    with pytest.raises(AnalysisError) as caught:
        m.causal_design_load(changed)
    assert caught.value.code == "invalid_saved_result"
    modified = copy.deepcopy(result)
    modified.attrs["state"] = state
    modified.attrs["state_sha256"] = m.digest(state)
    with pytest.raises(AnalysisError):
        m.causal_design_save(modified)


def test_all_eight_prespecified_support_values_full_grid_certificates_and_lp_oracle():
    support = [0, 1, 2, 4, 8, 16, 32, 64]
    c0, c1 = [1] * 8, [1, 2, 0, 1, 0, 1, 1, 1]
    result = quantile(
        data_for(support, c0, c1),
        support=support,
        quantiles=[0.25, 0.5, 0.75],
        max_work=1_000_000_000,
    )
    state = result.attrs["state"]
    mass = sum(c0) * sum(c1)
    differences = np.array([[b - a for b in support] for a in support])
    assert state["effect_grid"] == sorted(set(differences.flatten()))
    assert len(state["effect_grid"]) <= 64
    for record in state["extrema"]:
        np.testing.assert_allclose(
            np.array([record["lower_mass_count"], record["upper_mass_count"]]) / mass,
            transport_lp(c0, c1, support, record["threshold"]),
            atol=2e-15,
        )
        for bound in ("lower", "upper"):
            mask = differences <= record["threshold"]
            assert_certificate(
                record[bound],
                np.array(c0) * sum(c1),
                np.array(c1) * sum(c0),
                mask if bound == "upper" else ~mask,
            )
    assert result.attrs["state"]["sample"]["planned_work"] <= 1_000_000_000


def test_unobserved_prespecified_support_cells_retained_and_extreme_translation_scaling():
    base = data_for([0, 1, 2], [1, 1, 0], [0, 1, 2])
    a = quantile(base, support=[0, 1, 2, 3], quantiles=[0.25, 0.5, 0.75])
    assert a.attrs["state"]["arm_counts"] == [[1, 1, 0, 0], [0, 1, 2, 0]]
    assert a.attrs["state"]["denominator"] == 6
    translated = base.copy()
    translated["y"] = translated.y.astype(float) + 2**40
    b = quantile(translated, support=[2**40 + x for x in range(4)], quantiles=[0.25, 0.5, 0.75])
    pd.testing.assert_frame_equal(a["bounds"], b["bounds"], check_exact=True)
    scaled = base.copy()
    scale = math.ldexp(1.0, 490)
    scaled["y"] = scaled.y.astype(float) * scale
    c = quantile(scaled, support=[x * scale for x in range(4)], quantiles=[0.25, 0.5, 0.75])
    for column in ("lower_bound", "upper_bound"):
        np.testing.assert_array_equal(c["bounds"][column].to_numpy() / scale, a["bounds"][column])


def test_grid_exact_conversion_preserves_original_scalar_values():
    for keyword in ("support", "thresholds"):
        with pytest.raises(AnalysisError, match="changes during float64"):
            cdf(**{keyword: [Fraction(1, 3)]})
    with pytest.raises(AnalysisError, match="changes during float64"):
        quantile(quantiles=[Fraction(1, 3)])
    assert (
        cdf(support=[Fraction(0), Fraction(1)], thresholds=[Fraction(0)])["bounds"].threshold.iloc[
            0
        ]
        == 0
    )


def test_native_float_precision_branch_refuses_extended_domain_or_accepts_binary64(monkeypatch):
    precise = np.nextafter(np.longdouble(1), np.longdouble(2))
    data = pd.DataFrame(dict(y=np.array([precise, 1], dtype=np.longdouble), d=[0, 1]))
    if data.y.dtype.itemsize > 8:

        def forbidden(*args, **kwargs):
            pytest.fail("No sample conversion is allowed for an extended floating domain")

        with monkeypatch.context() as local:
            local.setattr(m, "sample", forbidden)
            with pytest.raises(AnalysisError) as caught:
                cdf(data, support=[1])
            assert caught.value.code == "unsupported_dtype"
    else:
        result = cdf(data, support=[1, precise], thresholds=[0])
        assert result.attrs["state"]["outcomes"] == [float(precise), 1]
    for value, name in (
        (precise, "support"),
        (precise, "thresholds"),
        (np.nextafter(np.longdouble(0.5), np.longdouble(1)), "quantiles"),
    ):
        if value != np.longdouble(float(value)):
            with pytest.raises(AnalysisError, match="changes during float64"):
                e._grid([value], name, 8, quantiles=name == "quantiles")
        else:
            assert e._grid([value], name, 8, quantiles=name == "quantiles") == [float(value)]
