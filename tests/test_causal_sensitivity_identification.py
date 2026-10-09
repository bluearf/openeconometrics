"""Primary-formula, exhaustive-completion and independent LP bound oracles."""

import copy
import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.econometrics.causal_design.partial_identification import evalue, lee_bounds, manski_ate
from openecon.resources import use_workspace_budget


def _lee_data(decreasing=False, ties=True):
    # Different arm counts deliberately produce 3.5 retained high-arm rows,
    # requiring a fractional boundary rather than selecting an integer subset.
    data = pd.DataFrame({
        "y": [0., 3., np.nan, np.nan, 1., 2., 2., 2., 10., np.nan, np.nan],
        "d": [0] * 4 + [1] * 7,
        "s": [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    }, index=["duplicate", 1, "1", True, *range(7)])
    if not ties:
        data.iloc[5:8, 0] = [2., 4., 6.]
    if decreasing:
        data.d = 1 - data.d
    return data


def _lee(data=None, **options):
    return lee_bounds(_lee_data() if data is None else data, "y", "d", "s",
                      **({"design": "randomized", "monotonicity": "increasing"} | options))


def _manski(data=None, **options):
    if data is None:
        data = pd.DataFrame({"y": [0., .3, .7, 1., .2], "d": [0, 0, 1, 1, 1]})
    return manski_ate(data, "y", "d", **({"lower": 0., "upper": 1.} | options))


def test_evalue_primary_author_published_reference_and_confidence_limit():
    # Published author-package Hammond/Horn example, direct RR route only.
    data = pd.DataFrame({"rr": [10.73], "lo": [8.02], "hi": [14.36]})
    output = evalue(data, "rr", lower="lo", upper="hi")
    row = output["evalues"].iloc[0]
    assert row.e_value == pytest.approx(20.94777, abs=5e-6)
    assert row.e_value_ci == pytest.approx(15.52336, abs=5e-6)
    assert row.ci_limit_closest_to_null == 8.02
    assert output.attrs["n"] == 1
    assert output.attrs["state"]["covariance"] is None


def test_evalue_independent_quadratic_bias_factor_oracle_protective_and_null_ci():
    data = pd.DataFrame({
        "rr": [2., .25, 1., 1.5, .5],
        "lo": [1.2, .1, .7, .8, .3], "hi": [3., .5, 1.3, 2.2, 1.2],
    })
    output = evalue(data, "rr", lower="lo", upper="hi")
    table = output["evalues"]
    # E is the larger root of E^2 - 2*R*E + R = 0, independently
    # evaluated through NumPy polynomial roots rather than the implementation.
    for i, rr in enumerate(data.rr):
        transformed = rr if rr >= 1 else 1 / rr
        expected = 1.0 if transformed == 1 else max(np.roots([1., -2 * transformed, transformed]))
        assert table.e_value.iloc[i] == pytest.approx(expected, abs=2e-14)
        e = table.e_value.iloc[i]
        assert e * e / (2 * e - 1) == pytest.approx(transformed, abs=2e-14)
    assert list(table.e_value_ci.iloc[2:]) == [1., 1., 1.]
    assert table.ci_limit_closest_to_null.iloc[1] == .5
    assert table.ci_rr_away_from_null.iloc[1] == 2.
    assert table.e_value.iloc[2] == 1.
    assert list(table.direction) == ["increased_risk", "protective_inverse", "null", "increased_risk", "protective_inverse"]


def test_evalue_no_limits_are_undefined_and_reciprocal_symmetric():
    data = pd.DataFrame({"rr": [2., .5, 10., .1]})
    output = evalue(data, "rr")
    np.testing.assert_allclose(output["evalues"].e_value.iloc[::2], output["evalues"].e_value.iloc[1::2])
    assert output["evalues"].e_value_ci.isna().all()
    assert output.attrs["state"]["confidence_interval"] is None
    assert output.attrs["state"]["confidence_limit_evalues"] is None
    with pytest.raises(TypeError):
        evalue(data, "rr", measure="odds_ratio")


def test_evalue_small_protective_ratio_stable_arithmetic_and_overflow_refusal():
    output = evalue(pd.DataFrame({"rr": [1e-300]}), "rr")
    assert output["evalues"].e_value.iloc[0] == pytest.approx(2e300)
    with pytest.raises(AnalysisError, match="overflow|finite"):
        evalue(pd.DataFrame({"rr": [1e-320]}), "rr")


@pytest.mark.parametrize("data,options", [
    (pd.DataFrame({"rr": [0., 1.]}), {}),
    (pd.DataFrame({"rr": [-1., 1.]}), {}),
    (pd.DataFrame({"rr": [np.inf, 1.]}), {}),
    (pd.DataFrame({"rr": ["2", "1"]}), {}),
    (pd.DataFrame({"rr": [True]}), {}),
    (pd.DataFrame({"rr": [2.], "lo": [1.]}), {"lower": "lo"}),
    (pd.DataFrame({"rr": [2.], "hi": [3.]}), {"upper": "hi"}),
    (pd.DataFrame({"rr": [2.], "lo": [3.], "hi": [4.]}), {"lower": "lo", "upper": "hi"}),
    (pd.DataFrame({"rr": [2.], "lo": [0.], "hi": [3.]}), {"lower": "lo", "upper": "hi"}),
    (pd.DataFrame({"rr": [2.]}), {"lower": "rr", "upper": "rr"}),
])
def test_evalue_invalid_roles_or_rr_limits_fail_closed(data, options):
    with pytest.raises(AnalysisError):
        evalue(data, "rr", **options)


def test_manski_exhaustive_individual_counterfactual_completion_sharpness():
    data = pd.DataFrame({"y": [.2, .5, .8, .9, .1], "d": [0, 0, 0, 1, 1]})
    output = _manski(data, lower=-.5, upper=1.5)
    effect_extremes = []
    for counterfactuals in itertools.product([-.5, 1.5], repeat=len(data)):
        effect = [observed - missing if d else missing - observed
                  for observed, d, missing in zip(data.y, data.d, counterfactuals)]
        effect_extremes.append(np.mean(effect))
    row = output["bounds"].iloc[0]
    assert row.lower == pytest.approx(min(effect_extremes), abs=1e-15)
    assert row.upper == pytest.approx(max(effect_extremes), abs=1e-15)
    assert row.width == pytest.approx(2.)
    # Saved extremizers actually attain both identified endpoints.
    extreme = output["extremizers"]
    assert np.mean(extreme.y1_lower - extreme.y0_upper) == pytest.approx(row.lower)
    assert np.mean(extreme.y1_upper - extreme.y0_lower) == pytest.approx(row.upper)
    assert output.attrs["state"]["arm_probabilities"] == [.6, .4]
    assert output.attrs["state"]["observed_joint_moments"] == pytest.approx([.3, .2])


def test_manski_joint_moments_do_not_replace_the_target_with_conditional_difference():
    data = pd.DataFrame({"y": [0., 0., 0., 0., 1.], "d": [0, 0, 0, 0, 1]})
    row = _manski(data)["bounds"].iloc[0]
    assert row.lower == pytest.approx(0.)
    assert row.upper == pytest.approx(1.)
    # A raw between-arm difference is 1, whereas consistency alone allows 0.
    assert row.upper - row.lower == pytest.approx(1.)


def test_manski_affine_support_transformation_and_singleton_support():
    data = pd.DataFrame({"y": [.1, .3, .5, .9], "d": [0, 0, 1, 1]})
    base = _manski(data)["bounds"].iloc[0]
    affine = _manski(data.assign(y=10 + 3 * data.y), lower=10, upper=13)["bounds"].iloc[0]
    np.testing.assert_allclose([affine.lower, affine.upper], 3 * np.array([base.lower, base.upper]), atol=2e-15)
    singleton = _manski(data.assign(y=5.), lower=5., upper=5.)
    assert list(singleton["bounds"].iloc[0][["lower", "upper", "width"]]) == [0., 0., 0.]
    np.testing.assert_array_equal(singleton.attrs["state"]["potential_mean_bounds"], [[5., 5.], [5., 5.]])


@pytest.mark.parametrize("options", [{"lower": 2.}, {"lower": np.nan}, {"upper": np.inf},
                                    {"lower": True}, {"upper": .5}])
def test_manski_invalid_support_and_observed_violations(options):
    with pytest.raises(AnalysisError):
        _manski(**options)


def test_manski_numeric_underflow_and_unrepresentable_width_fail_closed():
    with pytest.raises(AnalysisError, match="underflow"):
        _manski(pd.DataFrame({"y": [0., 0.], "d": [0, 1]}), lower=0., upper=5e-324)
    with pytest.raises(AnalysisError, match="finite"):
        _manski(lower=-1e308, upper=1e308)


@pytest.mark.parametrize("decreasing", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_lee_independent_linear_program_tail_oracle_fractional_mass(decreasing, ties):
    data = _lee_data(decreasing=decreasing, ties=ties)
    output = _lee(data, monotonicity="decreasing" if decreasing else "increasing")
    high_arm = 0 if decreasing else 1
    high = data.loc[(data.d == high_arm) & (data.s == 1), "y"].to_numpy()
    low = data.loc[(data.d != high_arm) & (data.s == 1), "y"].to_numpy()
    retain_mass = 3.5
    constraint = np.ones((1, len(high)))
    minimum = linprog(high, A_eq=constraint, b_eq=[retain_mass], bounds=[(0, 1)] * len(high), method="highs")
    maximum = linprog(-high, A_eq=constraint, b_eq=[retain_mass], bounds=[(0, 1)] * len(high), method="highs")
    assert minimum.success and maximum.success
    low_mean, high_mean = minimum.fun / retain_mass, -maximum.fun / retain_mass
    expected = [low.mean() - high_mean, low.mean() - low_mean] if decreasing else [low_mean - low.mean(), high_mean - low.mean()]
    row = output["bounds"].iloc[0]
    np.testing.assert_allclose([row.lower, row.upper], expected, atol=1e-14)
    state = output.attrs["state"]
    assert state["retained_selected_mass"] == {"numerator": 14, "denominator": 4, "value": 3.5}
    assert state["trimmed_arm"] == high_arm
    assert state["retention_fraction"] == pytest.approx(.7)
    assert state["always_selected_fraction"] == .5
    # Every arm weight and every unselected physical row is persisted.
    table = output["trimming_weights"]
    assert len(table) == len(data) and list(table.position) == list(range(len(data)))
    assert (table.loc[data.s.to_numpy() == 0, "retention_lower_bound"] == 0).all()
    for key in ("mean_weight_lower_bound", "mean_weight_upper_bound"):
        for arm in (0, 1):
            assert table.loc[data.d.to_numpy() == arm, key].sum() == pytest.approx(1.)
    np.testing.assert_allclose(np.nansum(np.where(data.d == 1, 1, -1) * table.mean_weight_lower_bound * data.y.to_numpy()), row.lower)
    assert state["standard_error"] is None and state["confidence_interval"] is None


def test_lee_boundary_ties_share_mass_and_permutation_equivariance():
    data = _lee_data()
    output = _lee(data)
    table = output["trimming_weights"]
    tied = np.array([5, 6, 7])
    np.testing.assert_allclose(table.retention_lower_bound.iloc[tied], [5 / 6] * 3)
    np.testing.assert_allclose(table.retention_upper_bound.iloc[tied], [5 / 6] * 3)
    permutation = np.random.default_rng(8).permutation(len(data))
    permuted = _lee(data.iloc[permutation])
    np.testing.assert_allclose(permuted["bounds"].iloc[0][["lower", "upper"]].to_numpy(dtype=float),
                               output["bounds"].iloc[0][["lower", "upper"]].to_numpy(dtype=float), atol=1e-14)
    np.testing.assert_allclose(permuted["trimming_weights"].retention_lower_bound.to_numpy()[np.argsort(permutation)],
                               table.retention_lower_bound, atol=1e-15)


def test_lee_equal_selection_rates_no_trimming_and_single_selected_each_arm():
    data = pd.DataFrame({"y": [1., np.nan, 4., np.nan], "d": [0, 0, 1, 1], "s": [1, 0, 1, 0]})
    for direction in ("increasing", "decreasing"):
        output = _lee(data, monotonicity=direction)
        assert list(output["bounds"].iloc[0][["lower", "upper", "width"]]) == [3., 3., 0.]
        assert output.attrs["state"]["retention_fraction"] == 1.
    no_attrition = data.assign(y=[1., 2., 3., 4.], s=1)
    output = _lee(no_attrition)
    assert list(output["bounds"].iloc[0][["lower", "upper"]]) == [2., 2.]


def test_lee_negative_affine_transform_swaps_bound_endpoints():
    data = _lee_data()
    original = _lee(data)["bounds"].iloc[0]
    transformed = _lee(data.assign(y=8 - 2 * data.y))["bounds"].iloc[0]
    np.testing.assert_allclose([transformed.lower, transformed.upper], [-2 * original.upper, -2 * original.lower], atol=1e-14)


def test_lee_all_original_rows_define_rates_and_ignored_outcomes_never_imputed():
    data = _lee_data()
    output = _lee(data)
    filled = data.copy()
    filled.loc[filled.s == 0, "y"] = 1e100
    other = _lee(filled)
    pd.testing.assert_frame_equal(output["bounds"], other["bounds"], check_exact=True)
    assert output.attrs["positions"] == list(range(11))
    assert output.attrs["state"]["arm_counts"] == [4, 7]
    assert output.attrs["state"]["selected_counts"] == [2, 5]
    assert output.attrs["n_selected"] == 7 and output.attrs["n_unselected"] == 4
    assert output.attrs["outcome_observation_sha256"] == other.attrs["outcome_observation_sha256"]
    assert causal_design_save(output) == causal_design_save(other)
    assert output.attrs["state"]["observed_y"] == other.attrs["state"]["observed_y"]
    assert output.attrs["unit_labels"] == list(data.index)


@pytest.mark.parametrize("change", ["broken_assignment", "missing_selection", "nonbinary_selection", "selected_missing", "selected_inf", "empty_control", "boolean_selection"])
def test_lee_topology_and_selected_outcome_guards(change):
    data = _lee_data()
    if change == "broken_assignment":
        data.iloc[2, data.columns.get_loc("d")] = np.nan
    elif change == "missing_selection":
        data.iloc[2, data.columns.get_loc("s")] = np.nan
    elif change == "nonbinary_selection":
        data.iloc[2, data.columns.get_loc("s")] = 2
    elif change == "selected_missing":
        data.iloc[0, 0] = np.nan
    elif change == "selected_inf":
        data.iloc[0, 0] = np.inf
    elif change == "empty_control":
        data.loc[data.d == 0, "s"] = 0
    elif change == "boolean_selection":
        data.s = data.s.astype(bool)
    with pytest.raises(AnalysisError):
        _lee(data)


def test_lee_direction_and_design_must_be_declared_and_are_not_inferred():
    data = _lee_data()
    with pytest.raises(TypeError):
        lee_bounds(data, "y", "d", "s")
    with pytest.raises(TypeError):
        lee_bounds(data, "y", "d", "s", design="randomized")
    with pytest.raises(AnalysisError, match="direction"):
        _lee(data, monotonicity="decreasing")
    for options in ({"design": "observational"}, {"monotonicity": "auto"}, {"missing": "drop"}):
        with pytest.raises(AnalysisError):
            _lee(data, **options)


def test_lee_numeric_underflow_is_not_reported_as_zero_trimmed_mean():
    data = pd.DataFrame({"y": [0., 0., 5e-324, 5e-324], "d": [0, 0, 1, 1], "s": [1, 1, 1, 1]})
    with pytest.raises(AnalysisError, match="underflow"):
        _lee(data)


@pytest.mark.parametrize("kind", ["evalue", "manski", "lee"])
def test_all_procedures_full_save_load_checksums_tables_dtypes_and_no_rng_mutation(tmp_path, kind):
    before = torch.random.get_rng_state().clone()
    if kind == "evalue":
        data = pd.DataFrame({"rr": [2., .5, 1.]}, index=["unit", 1, True])
        output = evalue(data, "rr")
    else:
        output = _manski() if kind == "manski" else _lee()
    assert torch.equal(before, torch.random.get_rng_state())
    path = tmp_path / f"{kind}.json"
    artifact = causal_design_save(output, path)
    restored = causal_design_load(path)
    assert artifact == causal_design_save(restored)
    assert list(restored) == list(output)
    for name in output:
        pd.testing.assert_frame_equal(restored[name], output[name], check_exact=True)
    tampered = copy.deepcopy(artifact)
    tampered["payload"]["attrs"]["state"]["inference"] = "fake confidence interval"
    with pytest.raises(AnalysisError, match="checksum"):
        causal_design_load(tampered)
    assert output.attrs["stata_parity_validated"] is False


@pytest.mark.parametrize("options", [{"device": "cuda"}, {"weights": "w"}, {"max_work": True}])
@pytest.mark.parametrize("kind", ["evalue", "manski", "lee"])
def test_unsupported_generic_options_rejected(kind, options):
    with pytest.raises(AnalysisError):
        if kind == "evalue":
            evalue(pd.DataFrame({"rr": [2.]}), "rr", **options)
        elif kind == "manski":
            _manski(**options)
        else:
            _lee(**options)


def test_summary_and_manski_explicit_drop_preserve_sample_positions():
    data = pd.DataFrame({"rr": [2., np.nan, .5]}, index=["dup", "dup", "dup"])
    output = evalue(data, "rr", missing="drop")
    assert output.attrs["positions"] == [0, 2]
    assert list(output["evalues"].position) == [0, 2]
    mdata = pd.DataFrame({"y": [0., np.nan, .8, 1.], "d": [0, 1, 1, 1]})
    output = _manski(mdata, missing="drop")
    assert output.attrs["positions"] == [0, 2, 3]
    assert "dropping" in output.attrs["state"]["missing_target"]


def test_full_work_and_workspace_preflight_no_partial_identification_results(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("No trim sort is permitted after failed preflight.")

    monkeypatch.setattr(torch, "argsort", refuse)
    for method in (lambda: evalue(pd.DataFrame({"rr": [2.]}), "rr", max_work=1),
                   lambda: _manski(max_work=1), lambda: _lee(max_work=1)):
        with pytest.raises(AnalysisError, match="max_work"):
            method()
    data = pd.DataFrame({"y": np.zeros(1500), "d": [0] * 750 + [1] * 750, "s": np.ones(1500)})
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        _lee(data)


def test_evalue_near_null_limit_is_not_clipped_to_one():
    ratio = math.nextafter(1., math.inf)
    output = evalue(pd.DataFrame({"rr": [ratio]}), "rr")
    assert output["evalues"].e_value.iloc[0] > 1.


@pytest.mark.parametrize("kind", ["evalue", "manski", "lee"])
def test_dataset_is_refused_without_consuming_replayable_input(kind):
    def refuse_collection():
        raise AssertionError("A resident-only bound must never collect this Dataset.")

    source = Dataset.from_batches(refuse_collection, columns=["rr", "y", "d", "s"], row_count=10)
    with pytest.raises(AnalysisError, match="resident"):
        if kind == "evalue":
            evalue(source, "rr")
        elif kind == "manski":
            _manski(source)
        else:
            _lee(source)


@pytest.mark.parametrize("kind", ["evalue", "manski", "lee"])
def test_explicit_cpu_route_overrides_global_default_device(kind):
    with torch.device("meta"):
        if kind == "evalue":
            output = evalue(pd.DataFrame({"rr": [2.]}), "rr")
        elif kind == "manski":
            output = _manski()
        else:
            output = _lee()
    assert output.attrs["device"] == "cpu"
    assert output.attrs["state"]["sample"]["dtype"] == "float64"


def test_lee_selected_outcome_or_original_position_changes_full_scientific_identity():
    data = _lee_data()
    original = _lee(data)
    changed = data.copy()
    changed.iloc[0, 0] += .1
    altered = _lee(changed)
    assert original.attrs["sample_sha256"] == altered.attrs["sample_sha256"]
    assert original.attrs["outcome_observation_sha256"] != altered.attrs["outcome_observation_sha256"]
    # Moving the selected observations preserves the target value but changes
    # the original-position identity instead of silently aligning by labels.
    reordered = _lee(data.iloc[[1, 0, *range(2, len(data))]])
    assert original.attrs["outcome_observation_sha256"] != reordered.attrs["outcome_observation_sha256"]
    pd.testing.assert_frame_equal(original["bounds"], reordered["bounds"], check_exact=True)
