"""Independent weighted moments, typed category identity and saved-schema guards."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.regularized.glm_design import fit_design, transform_design


def fit(df, predictors=None, categorical=None, weights=None, **options):
    settings = dict(intercept=True, standardize=True, penalty_factors=None, forced_controls=None)
    settings.update(options)
    return fit_design(
        df,
        list(df.columns) if predictors is None else predictors,
        [] if categorical is None else categorical,
        torch.ones(len(df), dtype=torch.float64) if weights is None else weights,
        **settings,
    )


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_weighted_design_agrees_with_independent_population_moments(intercept, standardize):
    values = np.array([[2., -1.], [-3., 5.], [8., 2.], [0., 4.]])
    df = pd.DataFrame(values, columns=["x", "u"])
    raw = np.array([1., 3., 2., 4.])
    z, state, factors = fit(df, weights=torch.tensor(raw), intercept=intercept, standardize=standardize)
    center = np.average(values, axis=0, weights=raw) if intercept else np.zeros(2)
    residual = values - center
    scale = np.sqrt(np.average(residual ** 2, axis=0, weights=raw)) if standardize else np.ones(2)
    np.testing.assert_allclose(state["centers"], center, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(state["scales"], scale, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z.numpy(), residual / scale, rtol=1e-14, atol=1e-14)
    assert z.dtype == torch.float64 and z.device.type == "cpu"
    assert factors.tolist() == [1., 1.]


@pytest.mark.parametrize("intercept", [False, True])
def test_frequency_weight_replication_and_raw_weight_scale_invariance(intercept):
    df = pd.DataFrame({"x": [-3., 2., 7., 4.], "g": ["b", "a", "c", "a"]})
    original = df.copy(deep=True)
    weights = torch.tensor([2., 3., 1., 4.], dtype=torch.float64)
    z, state, factors = fit(df, categorical=["g"], weights=weights, intercept=intercept)
    replicated = df.iloc[np.repeat(np.arange(4), weights.numpy().astype(int))]
    replicated_z, replicated_state, _ = fit(replicated, categorical=["g"], intercept=intercept)
    np.testing.assert_allclose(replicated_state["centers"], state["centers"], atol=1e-14)
    np.testing.assert_allclose(replicated_state["scales"], state["scales"], atol=1e-14)
    np.testing.assert_allclose(replicated_z.numpy(), z.numpy()[np.repeat(np.arange(4), weights.numpy().astype(int))], atol=1e-14)
    for multiplier in (1e-280, 1e280):
        rescaled_z, rescaled_state, rescaled_factors = fit(df, categorical=["g"], weights=weights * multiplier, intercept=intercept)
        np.testing.assert_allclose(rescaled_z.numpy(), z.numpy(), atol=1e-14)
        np.testing.assert_allclose(rescaled_state["centers"], state["centers"], atol=1e-14)
        np.testing.assert_allclose(rescaled_state["scales"], state["scales"], atol=1e-14)
        assert rescaled_factors.tolist() == factors.tolist()
    pd.testing.assert_frame_equal(original, df)


def test_weight_sum_overflow_is_avoided_and_extreme_predictor_moments_remain_finite():
    df = pd.DataFrame({"x": [1e308, -1e308, -1e308]})
    weights = torch.tensor([1e308, 1e308, 1e308], dtype=torch.float64)
    assert not torch.isfinite(weights.sum())
    z, state, _ = fit(df, weights=weights)
    assert all(np.isfinite(state[key]).all() for key in ("centers", "scales"))
    np.testing.assert_allclose(z.numpy().ravel(), [np.sqrt(2), -1 / np.sqrt(2), -1 / np.sqrt(2)], rtol=1e-14)
    np.testing.assert_allclose(transform_design(df, state).numpy(), z.numpy(), rtol=1e-14)


def test_transform_avoids_overflowing_centered_difference_when_standardized_value_is_finite():
    train = pd.DataFrame({"x": [-1e308, -1e308, 1e308]})
    z, state, _ = fit(train, weights=torch.tensor([9., 9., 2.], dtype=torch.float64))
    assert not np.isfinite(1e308 - state["centers"][0])
    replay = transform_design(train, state)
    np.testing.assert_allclose(replay.numpy(), z.numpy(), rtol=1e-14)


@pytest.mark.parametrize("intercept", [False, True])
def test_constant_numeric_and_one_level_categories_are_not_silently_dropped(intercept):
    df = pd.DataFrame({"x": [3., 3., 3.], "zero": [0., 0., 0.], "g": ["only"] * 3})
    z, state, factors = fit(df, categorical=["g"], weights=torch.tensor([1., 2., 7.]), intercept=intercept)
    assert state["terms"] == (["x", "zero"] if intercept else ["x", "zero", 'C(g)[str:"only"]'])
    assert state["scales"] == ([1., 1.] if intercept else [3., 1., 1.])
    np.testing.assert_allclose(z.numpy(), np.tile([0., 0.] if intercept else [1., 0., 1.], (3, 1)), atol=0)
    assert factors.numel() == len(state["terms"])
    torch.testing.assert_close(transform_design(df, state), z, rtol=0, atol=0)


def test_empty_predictors_produce_slope_free_state_and_replay():
    df = pd.DataFrame({"unused": [1., 2., 3.]})
    z, state, factors = fit(df, predictors=[])
    assert z.shape == (3, 0) and factors.shape == (0,)
    assert state["terms"] == state["centers"] == state["scales"] == []
    assert transform_design(df.iloc[:1], json.loads(json.dumps(state))).shape == (1, 0)


@pytest.mark.parametrize("intercept", [False, True])
def test_category_identity_is_typed_first_appearance_and_json_replayable(intercept):
    df = pd.DataFrame({"g": pd.Series([True, 1, "1", 1., False, 0, "0", 0., True, "1"], dtype=object)})
    z, state, _ = fit(df, categorical=["g"], intercept=intercept, standardize=False)
    expected = [
        {"type": "bool", "value": True}, {"type": "int", "value": 1},
        {"type": "str", "value": "1"}, {"type": "float", "value": 1.},
        {"type": "bool", "value": False}, {"type": "int", "value": 0},
        {"type": "str", "value": "0"}, {"type": "float", "value": 0.},
    ]
    assert state["levels"]["g"] == expected
    assert state["term_levels"] == expected[int(intercept):]
    assert len(set(state["terms"])) == 8 - int(intercept)
    raw = np.array([[type(value) is type(level["value"]) and value == level["value"] for level in expected[int(intercept):]] for value in df["g"]], dtype=float)
    np.testing.assert_allclose(z.numpy(), raw - raw.mean(0) if intercept else raw, atol=1e-14)
    restored = json.loads(json.dumps(state, allow_nan=False))
    torch.testing.assert_close(transform_design(df.iloc[::-1], restored), z.flip(0), rtol=1e-14, atol=1e-14)


def test_unused_declared_categorical_levels_are_excluded_from_training_schema():
    df = pd.DataFrame({"g": pd.Categorical(["b", "a", "b"], categories=["z", "a", "b", "unseen"])})
    _, state, _ = fit(df, categorical=["g"])
    assert state["levels"]["g"] == [{"type": "str", "value": "b"}, {"type": "str", "value": "a"}]
    with pytest.raises(AnalysisError, match="absent from its training schema") as caught:
        transform_design(pd.DataFrame({"g": ["z"]}), state)
    assert caught.value.code == "unseen_category"


def test_schema_is_training_only_and_query_rows_never_change_scales_or_category_order():
    train = pd.DataFrame({"x": [1., 3., 2.], "g": ["b", "a", "b"]})
    _, state, _ = fit(train, categorical=["g"])
    original_state = copy.deepcopy(state)
    query = pd.DataFrame({"x": [999., -50.], "g": ["a", "b"]}, index=[123, 123])
    replay = transform_design(query, state).numpy()
    raw = np.array([[999., 1.], [-50., 0.]])
    np.testing.assert_allclose(replay, (raw - state["centers"]) / state["scales"], rtol=1e-14)
    assert state == original_state
    with pytest.raises(AnalysisError) as caught:
        transform_design(pd.DataFrame({"x": [2.], "g": ["new"]}), state)
    assert caught.value.code == "unseen_category"


def test_predictor_factors_apply_to_entire_encoded_block_without_renormalization():
    df = pd.DataFrame({"x": [1., 2., 3., 4.], "g": ["b", "a", "c", "b"], "u": [2., 1., 5., 7.]})
    _, state, factors = fit(df, categorical=["g"], penalty_factors={"x": 3., "g": 7., "u": 0.})
    assert factors.tolist() == [3., 7., 7., 0.]
    assert state["effective_factors"] == factors.tolist()
    _, forced_state, forced = fit(df, categorical=["g"], penalty_factors={"x": 3., "g": 7.}, forced_controls=["g"])
    assert forced.tolist() == [3., 0., 0., 1.]
    assert forced_state["forced_controls"] == ["g"]
    assert forced_state["penalty_factors"] == {"x": 3., "g": 0., "u": 1.}
    assert transform_design(df, json.loads(json.dumps(forced_state))).shape == (4, 4)


@pytest.mark.parametrize("factors", [[], {"other": 1}, {"x": -1}, {"x": True}, {"x": "2"}, {"x": float("nan")}, {"x": float("inf")}, {"x": 10 ** 400}])
def test_bad_factor_config_fails_before_fit(factors):
    with pytest.raises(AnalysisError) as caught:
        fit(pd.DataFrame({"x": [1., 2.]}), penalty_factors=factors)
    assert caught.value.code == "invalid_penalty_factors"


@pytest.mark.parametrize("forced", ["x", ["absent"], ["x", "x"], [None]])
def test_bad_forced_controls_are_explicit_errors(forced):
    with pytest.raises(AnalysisError) as caught:
        fit(pd.DataFrame({"x": [1., 2.]}), forced_controls=forced)
    assert caught.value.code == "invalid_forced_controls"


@pytest.mark.parametrize("weights", [[1., 1.], torch.tensor([1.]), torch.tensor([[1., 1.]]), torch.tensor([True, True]), torch.tensor([1 + 0j, 1 + 0j]), torch.tensor([1., 0.]), torch.tensor([1., -1.]), torch.tensor([1., float("nan")]), torch.tensor([1., float("inf")])])
def test_invalid_weights_fail_without_silent_coercion(weights):
    with pytest.raises(AnalysisError) as caught:
        fit(pd.DataFrame({"x": [1., 2.]}), weights=weights)
    assert caught.value.code == "invalid_weights"


@pytest.mark.parametrize("bad", [None, pd.NA, pd.NaT, float("nan"), float("inf"), [1], {"a": 1}, pd.Timestamp("2020-01-01")])
def test_missing_nonfinite_or_non_json_category_values_fail_fit_and_transform(bad):
    train = pd.DataFrame({"g": ["a", "b"]})
    _, state, _ = fit(train, categorical=["g"])
    bad_frame = pd.DataFrame({"g": pd.Series([bad], dtype=object)})
    with pytest.raises(AnalysisError):
        fit(bad_frame, categorical=["g"])
    with pytest.raises(AnalysisError):
        transform_design(bad_frame, state)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "1", 1j])
def test_nonfinite_nonnumeric_or_complex_numeric_values_fail_fit_and_transform(bad):
    train = pd.DataFrame({"x": [1., 2.]})
    _, state, _ = fit(train)
    bad_frame = pd.DataFrame({"x": [bad]})
    with pytest.raises(AnalysisError):
        fit(bad_frame)
    with pytest.raises(AnalysisError):
        transform_design(bad_frame, state)


def test_width_guards_refuse_predictors_categories_and_expanded_designs():
    with pytest.raises(AnalysisError) as caught:
        fit(pd.DataFrame({f"x{i}": [1., 2.] for i in range(17)}))
    assert caught.value.code == "model_too_wide"
    with pytest.raises(AnalysisError) as caught:
        fit(pd.DataFrame({"g": list(range(17))}), categorical=["g"])
    assert caught.value.code == "model_too_wide"
    wide = pd.DataFrame({f"g{i}": list(range(16)) for i in range(5)})
    with pytest.raises(AnalysisError) as caught:
        fit(wide, categorical=list(wide.columns))
    assert caught.value.code == "model_too_wide"
    allowed = wide.iloc[:, :4]
    z, _, _ = fit(allowed, categorical=list(allowed.columns), intercept=False)
    assert z.shape == (16, 64)


@pytest.mark.parametrize("bad_frame", [pd.DataFrame({"other": [1.]}), pd.DataFrame([[1., 2.]], columns=["x", "x"]), pd.DataFrame({"x": []}), {"x": [1.]}])
def test_missing_duplicate_or_empty_columns_fail_replay(bad_frame):
    _, state, _ = fit(pd.DataFrame({"x": [1., 2.]}))
    with pytest.raises(AnalysisError):
        transform_design(bad_frame, state)


@pytest.mark.parametrize("change", [
    {"version": True}, {"version": 2}, {"predictors": ["g", "g"]}, {"categorical": ["absent"]},
    {"intercept": 1}, {"standardize": "yes"}, {"categorical_encoding": "one_hot"},
    {"levels": {"g": []}}, {"levels": {"g": [{"type": "int", "value": True}]}},
    {"levels": {"g": [{"type": "str", "value": "a"}] * 2}}, {"terms": ["broken"]},
    {"term_predictors": []}, {"term_levels": []}, {"centers": [float("nan")]},
    {"scales": [0.]}, {"scales": [-1.]}, {"scales": []}, {"effective_factors": [7.]}, {"effective_factors": [True]},
    {"penalty_factors": {"g": -1.}}, {"forced_controls": ["absent"]},
])
def test_invalid_saved_state_never_silently_reconstructs_a_different_design(change):
    df = pd.DataFrame({"g": ["a", "b", "a"]})
    _, state, _ = fit(df, categorical=["g"])
    state.update(change)
    with pytest.raises(AnalysisError) as caught:
        transform_design(df, state)
    assert caught.value.code == "invalid_prediction_state"


def test_unseen_equal_but_differently_typed_category_is_rejected():
    _, state, _ = fit(pd.DataFrame({"g": pd.Series([True, False], dtype=object)}), categorical=["g"])
    with pytest.raises(AnalysisError) as caught:
        transform_design(pd.DataFrame({"g": pd.Series([1], dtype=object)}), state)
    assert caught.value.code == "unseen_category"


def test_unstandardized_and_no_intercept_state_guards_are_consistent():
    df = pd.DataFrame({"x": [1., 2.]})
    _, state, _ = fit(df, intercept=False, standardize=False)
    for change in ({"centers": [1.]}, {"scales": [2.]}):
        broken = dict(state, **change)
        with pytest.raises(AnalysisError) as caught:
            transform_design(df, broken)
        assert caught.value.code == "invalid_prediction_state"


def test_literal_intercept_predictor_cannot_duplicate_generated_constant_term():
    df = pd.DataFrame({"Intercept": [1., 2., 3.]})
    with pytest.raises(AnalysisError) as caught:
        fit(df, intercept=True)
    assert caught.value.code == "duplicate_terms"


def test_literal_intercept_predictor_remains_available_without_generated_constant():
    df = pd.DataFrame({"Intercept": [1., 2., 3.]})
    z, state, factors = fit(df, intercept=False)
    assert state["terms"] == ["Intercept"] and state["centers"] == [0.]
    assert factors.tolist() == [1.]
    torch.testing.assert_close(transform_design(df, state), z)
