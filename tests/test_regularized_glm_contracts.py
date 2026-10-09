"""Public predictive GLM contracts, independent CV scoring and replay checks."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


FAMILIES = ("elasticnet_logit", "elasticnet_poisson")


def frame(family, n=72):
    rng = np.random.default_rng(5619)
    a = rng.normal(size=n)
    z = rng.normal(size=n)
    g = np.tile(["low", "middle", "high"], (n + 2) // 3)[:n]
    eta = -0.25 + 0.7 * a - 0.4 * z + 0.3 * (g == "high")
    y = (
        rng.binomial(1, 1 / (1 + np.exp(-eta)))
        if family == "elasticnet_logit"
        else rng.poisson(np.exp(eta))
    )
    return pd.DataFrame(
        {"y": y.astype(float), "a": a, "z": z, "g": g,
         "w": 1 + np.arange(n) % 4, "constant": 7.0},
        index=np.repeat(np.arange((n + 1) // 2)[::-1], 2)[:n],
    )


def fit(family, data=None, **options):
    return getattr(oe, family)(
        data=frame(family) if data is None else data,
        y="y", x=options.pop("x", ["a", "z"]),
        **({"selection": "fixed", "penalty": 0.12} | options),
    )


def state(result):
    return result.extra["regularized_glm_state"]


def prediction(result, data, **options):
    return np.asarray(oe.regularized_glm_predict(result, data, **options), dtype=float).reshape(-1)


def encoded(data, design):
    """Independently reconstruct saved scalar/dummy columns with NumPy."""
    columns = []
    for predictor, level in zip(design["term_predictors"], design["term_levels"], strict=True):
        values = data[predictor].to_numpy()
        columns.append(values.astype(float) if level is None else (values == level["value"]).astype(float))
    raw = np.column_stack(columns) if columns else np.empty((len(data), 0))
    return raw, (raw - np.asarray(design["centers"])) / np.asarray(design["scales"])


def observation_loss(eta, y, family):
    return np.logaddexp(0, eta) - y * eta if family == FAMILIES[0] else np.exp(eta) - y * eta


def resigned(result, *, allow_nan=False):
    """A checksum is not authentication: validate the admitted semantic domain."""
    s = state(result)
    encoded_state = json.dumps(
        {k: v for k, v in s.items() if k != "digest"},
        sort_keys=True, ensure_ascii=False, allow_nan=allow_nan, separators=(",", ":"),
    ).encode()
    s["digest"] = hashlib.sha256(encoded_state).hexdigest()
    return result


def assert_helpers_refuse(result, family):
    for helper in (lambda: oe.regularized_glm_path(result),
                   lambda: oe.regularized_glm_predict(result, frame(family))):
        with pytest.raises(AnalysisError) as caught:
            helper()
        assert caught.value.code == "invalid_result"


@pytest.mark.parametrize("family", FAMILIES)
def test_fixed_state_prediction_only_and_json_restoration(family):
    data = frame(family)
    original = data.copy(deep=True)
    result = fit(family, data)
    assert result.coefficients == [] and result.covariance_matrix == []
    assert result.inference["available"] is False
    assert result.inference["distribution"] == "none"
    assert result.extra["target"] == "prediction"
    assert result.sample_positions == list(range(len(data)))
    assert result.provenance["sample_hash"] and result.provenance["data_hash"]
    assert state(result)["digest"]
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored.model_dump() == result.model_dump()
    np.testing.assert_array_equal(prediction(restored, data), prediction(result, data))
    pd.testing.assert_frame_equal(oe.regularized_glm_path(restored), oe.regularized_glm_path(result))
    eta = prediction(result, data, kind="link")
    expected = 1 / (1 + np.exp(-eta)) if family == FAMILIES[0] else np.exp(eta)
    np.testing.assert_allclose(prediction(result, data), expected, rtol=1e-12, atol=1e-14)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("family", FAMILIES)
def test_weighted_fixed_fit_matches_literal_frequency_replication(family):
    data = frame(family, 48)
    weighted = fit(family, data, weights="w", weight_type="fweight")
    replicated = data.iloc[np.repeat(np.arange(len(data)), data.w.to_numpy())].reset_index(drop=True)
    expanded = fit(family, replicated)
    np.testing.assert_allclose(prediction(weighted, data), prediction(expanded, data), rtol=2e-7, atol=2e-8)
    assert weighted.nobs == len(data)
    assert weighted.sample_positions == list(range(len(data)))
    np.testing.assert_allclose(state(weighted)["raw_weights"], data.w)
    np.testing.assert_allclose(state(weighted)["normalized_weights"], data.w / data.w.sum())


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("weight_type", ["aweight", "pweight"])
def test_positive_weight_rescaling_leaves_prediction_objective_unchanged(family, weight_type):
    data = frame(family)
    model = fit(family, data, weights="w", weight_type=weight_type)
    rescaled = fit(family, data.assign(w=data.w * 1e150), weights="w", weight_type=weight_type)
    np.testing.assert_allclose(prediction(model, data), prediction(rescaled, data), rtol=2e-7, atol=2e-8)
    assert model.inference["available"] is False
    assert not model.tests


@pytest.mark.parametrize("family", FAMILIES)
def test_missing_weights_outcome_predictors_have_one_physical_sample(family):
    data = frame(family)
    data.iloc[1, data.columns.get_loc("y")] = np.nan
    data.iloc[3, data.columns.get_loc("a")] = np.nan
    data.iloc[7, data.columns.get_loc("w")] = np.nan
    result = fit(family, data, weights="w", missing="drop")
    keep = [i for i in range(len(data)) if i not in [1, 3, 7]]
    assert result.sample_positions == keep
    assert state(result)["physical_positions"] == keep
    assert result.nobs_original == len(data) and result.nobs == len(keep)
    assert result.dropped_rows == 3
    np.testing.assert_allclose(state(result)["raw_weights"], data.w.iloc[keep])
    with pytest.raises(AnalysisError):
        fit(family, data, weights="w", missing="raise")


@pytest.mark.parametrize("family", FAMILIES)
def test_predict_missing_preserves_duplicate_index_and_uses_no_outcome(family):
    model = fit(family)
    query = frame(family).iloc[[3, 3, 17, 21]][["a", "z"]].copy()
    query.iloc[1, 0] = np.nan
    with pytest.raises(AnalysisError):
        oe.regularized_glm_predict(model, query)
    out = oe.regularized_glm_predict(model, query, missing="drop")
    assert list(out.index) == list(query.index[[0, 2, 3]])
    np.testing.assert_allclose(np.asarray(out).reshape(-1), prediction(model, query.iloc[[0, 2, 3]]))


@pytest.mark.parametrize("family", FAMILIES)
def test_constant_column_and_forced_control_have_explicit_prediction_semantics(family):
    data = frame(family)
    plain = fit(family, data)
    constant = fit(family, data, x=["a", "z", "constant"])
    np.testing.assert_allclose(prediction(plain, data), prediction(constant, data), rtol=2e-7, atol=2e-8)
    forced = fit(family, data, penalty=10.0, forced_controls=["a"])
    penalized = fit(family, data, penalty=10.0)
    assert np.ptp(prediction(forced, data)) > 0.01
    assert np.ptp(prediction(penalized, data)) < 1e-6


@pytest.mark.parametrize("family", FAMILIES)
def test_categorical_saved_encoding_unknown_query_and_rare_cv_level_refusal(family):
    data = frame(family)
    model = fit(family, data, x=["a", "g"], categorical=["g"])
    query = data.iloc[[4, 1, 19]][["a", "g"]].copy()
    before = prediction(model, query)
    np.testing.assert_array_equal(before, prediction(model, query.copy()))
    with pytest.raises(AnalysisError):
        oe.regularized_glm_predict(model, query.assign(g="unseen"))
    rare = data.copy()
    rare.iloc[0, rare.columns.get_loc("g")] = "unique-validation-only"
    with pytest.raises(AnalysisError):
        fit(family, rare, x=["a", "g"], categorical=["g"], selection="cv",
            penalty=None, lambda_path=[0.3, 0.1], folds=3)


@pytest.mark.parametrize("family", FAMILIES)
def test_cv_seed_does_not_touch_global_numpy_or_torch_generator(family):
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    first = fit(family, selection="cv", penalty=None, lambda_path=[0.3, 0.1], folds=3, seed=981)
    second = fit(family, selection="cv", penalty=None, lambda_path=[0.3, 0.1], folds=3, seed=981)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert state(first)["fold_assignments"] == state(second)["fold_assignments"]
    assert state(first)["cv"] == state(second)["cv"]
    assert state(first)["paths"] == state(second)["paths"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_cv_complete_weighted_likelihood_and_training_only_design_oracle(family, automatic, standardize):
    data = frame(family)
    # Deliberately unequal fold weights and duplicate index labels test pooled
    # physical-row scoring, not an accidental equal mean of fold means.
    data["w"] = 1.0 + (np.arange(len(data)) % 7) ** 2
    result = fit(family, data, selection="cv", penalty=None,
                 lambda_path=None if automatic else [0.3, 0.1, 0.025],
                 n_lambdas=3, lambda_ratio=0.1, folds=3,
                 weights="w", standardize=standardize,
                 x=["a", "g"], categorical=["g"],
                 penalty_factors={"a": 2.0, "g": 0.5})
    s = state(result)
    score = np.zeros(3)
    seen = []
    assert len(s["fold_assignments"]) == len(data)
    assert len(s["cv"]) == 3
    for fold in s["cv"]:
        train = data.iloc[fold["training_positions"]]
        valid = data.iloc[fold["validation_positions"]]
        seen.extend(fold["validation_positions"])
        assert not set(fold["training_positions"]) & set(fold["validation_positions"])
        assert set(fold["training_positions"]) | set(fold["validation_positions"]) == set(range(len(data)))
        assert fold["design"]["levels"]["g"] == [
            {"type": "str", "value": v} for v in dict.fromkeys(train.g)
        ]
        raw, z = encoded(train, fold["design"])
        weights = train.w.to_numpy() / train.w.sum()
        centers = weights @ raw
        scales = np.sqrt(weights @ ((raw - centers) ** 2)) if standardize else np.ones(raw.shape[1])
        scales = np.where(scales == 0, 1.0, scales)
        np.testing.assert_allclose(fold["design"]["centers"], centers, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(fold["design"]["scales"], scales, rtol=1e-12, atol=1e-14)
        if automatic:
            mean = weights @ train.y.to_numpy()
            factors = np.asarray(fold["design"]["effective_factors"])
            score_at_null = np.abs(z.T @ (weights * (train.y.to_numpy() - mean)))
            reference = max(np.max(score_at_null / factors) / 0.5, 1e-8)
            np.testing.assert_allclose(fold["lambda_path"], reference * np.asarray(s["fractions"]), rtol=1e-12, atol=1e-14)
        else:
            assert fold["lambda_path"] == [0.3, 0.1, 0.025]
        _, query = encoded(valid, fold["design"])
        for j, point in enumerate(fold["paths"]):
            eta = query @ np.asarray(point["coefficients"]) + point["constant"]
            losses = observation_loss(eta, valid.y.to_numpy(), family)
            score[j] += np.dot(valid.w, losses) / data.w.sum()
            np.testing.assert_allclose(fold["validation_loss"][j], np.dot(valid.w, losses) / valid.w.sum(), rtol=1e-12, atol=1e-12)
    assert sorted(seen) == list(range(len(data)))
    np.testing.assert_allclose(s["cv_scores"], score, rtol=1e-12, atol=1e-12)
    expected = min(range(3), key=lambda i: (score[i], -s["lambda_path"][i]))
    assert s["selected_index"] == expected


@pytest.mark.parametrize("family", FAMILIES)
def test_heldout_outcomes_cannot_set_their_training_fold_auto_grid_or_fit(family):
    data = frame(family)
    settings = dict(selection="cv", penalty=None, n_lambdas=3, lambda_ratio=0.1,
                    folds=3, seed=982, weights="w")
    original = fit(family, data, **settings)
    fold = state(original)["cv"][1]
    heldout = fold["validation_positions"]
    altered = data.copy(deep=True)
    altered.iloc[heldout, altered.columns.get_loc("y")] = (
        1 - altered.y.iloc[heldout] if family == FAMILIES[0]
        else altered.y.iloc[heldout] + 5
    )
    changed = fit(family, altered, **settings)
    changed_fold = state(changed)["cv"][1]
    assert state(original)["fold_assignments"] == state(changed)["fold_assignments"]
    for field in ("training_positions", "validation_positions", "design", "paths", "lambda_path"):
        assert fold[field] == changed_fold[field]
    assert fold["validation_loss"] != changed_fold["validation_loss"]


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_cv_tie_chooses_largest_lambda_and_fweight_rows_stay_whole(family):
    data = frame(family)
    model = fit(family, data, x=["constant"], selection="cv", penalty=None,
                lambda_path=[0.1, 0.3, 0.2], folds=3,
                weights="w", weight_type="fweight")
    s = state(model)
    assert len(s["fold_assignments"]) == len(data) < data.w.sum()
    assert len({v for fold in s["cv"] for v in fold["validation_positions"]}) == len(data)
    np.testing.assert_array_equal(s["cv_scores"], np.repeat(s["cv_scores"][0], 3))
    assert s["selected_index"] == 0 and s["lambda_path"][0] == 0.3


@pytest.mark.parametrize("family", FAMILIES)
def test_cv_missing_sample_fold_ids_remain_original_physical_positions(family):
    data = frame(family)
    missing = [2, 7, 11]
    for row, name in zip(missing, ["y", "a", "w"], strict=True):
        data.iloc[row, data.columns.get_loc(name)] = np.nan
    model = fit(family, data, weights="w", selection="cv", penalty=None,
                lambda_path=[0.3, 0.1], folds=3, missing="drop")
    keep = [i for i in range(len(data)) if i not in missing]
    assert model.sample_positions == keep
    assert len(state(model)["fold_assignments"]) == len(keep)
    assert sorted(v for fold in state(model)["cv"] for v in fold["validation_positions"]) == keep
    assert all(set(fold["training_positions"]) | set(fold["validation_positions"]) == set(keep)
               for fold in state(model)["cv"])


@pytest.mark.parametrize("family", FAMILIES)
def test_dataset_refusal_happens_without_iteration(family, monkeypatch):
    data = Dataset.from_frame(frame(family))

    def forbidden(*args, **kwargs):
        raise AssertionError("An unsupported Dataset must not be collected or iterated")

    monkeypatch.setattr(data, "iter_batches", forbidden)
    with pytest.raises(AnalysisError) as caught:
        fit(family, data)
    assert caught.value.code == "streaming_unsupported"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("options", [
    {"device": "cuda"}, {"device": "mps"}, {"device": "auto"},
    {"selection": "plugin"}, {"selection": "fixed", "penalty": None},
    {"penalty": -1}, {"penalty": float("nan")}, {"penalty": True},
    {"lambda_path": []}, {"lambda_path": [0.1, 0.1]},
    {"lambda_path": [1, 0.1]}, {"lambda_path": [float("inf"), 0.12]},
    {"l1_ratio": -0.1}, {"l1_ratio": 1.1}, {"l1_ratio": True},
    {"max_work": 0}, {"max_work": True}, {"max_work": 1.5},
    {"tolerance": 0}, {"tolerance": float("nan")},
    {"max_iterations": 0}, {"standardize": "yes"},
    {"penalty_factors": {"unknown": 1}}, {"penalty_factors": {"a": -1}},
    {"forced_controls": ["unknown"]}, {"forced_controls": ["a", "a"]},
])
def test_invalid_controls_fail_without_coercion_or_silent_defaults(family, options):
    with pytest.raises((AnalysisError, ValidationError, TypeError)):
        fit(family, **options)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("column,value,weight_type", [
    ("w", -1.0, "aweight"), ("w", -1.0, "pweight"),
    ("w", 1.5, "fweight"), ("w", float("inf"), "aweight"),
])
def test_invalid_weights_refused(family, column, value, weight_type):
    data = frame(family)
    data[column] = data[column].astype(float)
    data.iloc[0, data.columns.get_loc(column)] = value
    with pytest.raises((AnalysisError, ValidationError)):
        fit(family, data, weights="w", weight_type=weight_type)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("weight_type", ["fweight", "aweight", "pweight"])
def test_zero_raw_weights_explicitly_exclude_physical_rows(family, weight_type):
    data = frame(family)
    zero = [1, 5, 23]
    data.iloc[zero, data.columns.get_loc("w")] = 0
    result = fit(family, data, weights="w", weight_type=weight_type)
    keep = [i for i in range(len(data)) if i not in zero]
    assert result.sample_positions == keep
    assert state(result)["physical_positions"] == keep
    assert result.nobs == len(keep) and result.dropped_rows == len(zero)
    assert all(v > 0 for v in state(result)["raw_weights"])
    np.testing.assert_allclose(state(result)["normalized_weights"], data.w.iloc[keep] / data.w.sum())
    explicit = fit(family, data.iloc[keep].reset_index(drop=True), weights="w", weight_type=weight_type)
    np.testing.assert_allclose(prediction(result, data), prediction(explicit, data), rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("value", [-0.1, 0.25, float("inf")])
def test_outcome_domain_is_enforced(family, value):
    data = frame(family)
    data.iloc[0, data.columns.get_loc("y")] = value
    with pytest.raises(AnalysisError):
        fit(family, data)


@pytest.mark.parametrize("family", FAMILIES)
def test_work_and_workspace_refused_before_solver(family):
    with pytest.raises(AnalysisError) as caught:
        fit(family, max_work=1)
    assert caught.value.code == "work_limit"
    big = frame(family, 4900)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as caught:
            fit(family, big, max_work=100000000000000)
    assert caught.value.code == "workspace_limit"
    assert caught.value.resource_plan["estimated_workspace_bytes"] > 1024**2


@pytest.mark.parametrize("family", FAMILIES)
def test_nonconvergence_cannot_be_saved_as_success(family):
    with pytest.raises(AnalysisError) as caught:
        fit(family, penalty=0.001, max_iterations=1, tolerance=1e-12)
    assert caught.value.code == "nonconvergence"


@pytest.mark.parametrize("family", FAMILIES)
def test_saved_state_corruption_and_rebinding_refused(family):
    result = fit(family)
    mutated = result.model_copy(deep=True)
    state(mutated)["selected_index"] = 99999
    with pytest.raises(AnalysisError):
        oe.regularized_glm_predict(mutated, frame(family))
    with pytest.raises(AnalysisError):
        oe.regularized_glm_path(mutated)
    rebound = result.model_copy(deep=True)
    rebound.spec = rebound.spec.model_copy(update={"predictors": ["z", "a"]})
    with pytest.raises(AnalysisError):
        oe.regularized_glm_predict(rebound, frame(family))
    foreign = result.model_copy(deep=True)
    foreign.spec = foreign.spec.model_copy(update={"estimator": FAMILIES[1] if family == FAMILIES[0] else FAMILIES[0]})
    with pytest.raises(AnalysisError):
        oe.regularized_glm_path(foreign)


@pytest.mark.parametrize("family", FAMILIES)
def test_prediction_work_device_and_kind_contracts(family):
    result = fit(family)
    for options in ({"kind": "class"}, {"kind": "bad"}, {"device": "cuda"},
                    {"max_work": 1}, {"max_work": True}, {"missing": "ignore"}):
        with pytest.raises((AnalysisError, ValidationError, TypeError)):
            oe.regularized_glm_predict(result, frame(family), **options)


@pytest.mark.parametrize("family", FAMILIES)
def test_unsupported_modelspec_covariance_formula_panel_are_rejected(family):
    data = frame(family)
    for extra in ({"covariance": "HC1"}, {"panel": "g"}, {"formula": "y ~ a + z"},
                  {"options": {"selection": "fixed", "penalty": 0.12, "offset": "a"}}):
        base = {"estimator": family, "outcome": "y", "predictors": ["a", "z"],
                "options": {"selection": "fixed", "penalty": 0.12}}
        with pytest.raises((AnalysisError, ValidationError, TypeError)):
            oe.fit(ModelSpec(**(base | extra)), data=data)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dimension", ["rows", "predictors", "categories", "path"])
def test_declared_resident_dimensions_are_never_truncated(family, dimension):
    data = frame(family, 5001 if dimension == "rows" else 72)
    options = {"max_work": 100000000000000}
    if dimension == "predictors":
        names = [f"feature_{i}" for i in range(17)]
        for i, name in enumerate(names):
            data[name] = data.a + i * data.z
        options["x"] = names
    elif dimension == "categories":
        data["g"] = [f"level_{i}" for i in range(len(data))]
        options.update(x=["g"], categorical=["g"])
    elif dimension == "path":
        options["lambda_path"] = [1.0] + list(np.linspace(0.5, 0.12, 100))
    with pytest.raises(AnalysisError):
        fit(family, data, **options)


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_penalty_factor_equals_forced_predictor_and_no_intercept_constant(family):
    data = frame(family)
    forced = fit(family, data, forced_controls=["a"])
    factor = fit(family, data, penalty_factors={"a": 0.0})
    np.testing.assert_allclose(prediction(forced, data), prediction(factor, data), rtol=2e-7, atol=2e-8)
    constant = fit(family, data, x=["a", "constant"], intercept=False,
                   forced_controls=["constant"], penalty=10.0)
    np.testing.assert_allclose(prediction(constant, data), data.y.mean(), rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("family", FAMILIES)
def test_unidentified_intercept_only_outcome_does_not_fake_finite_success(family):
    data = frame(family)
    for value in ([0.0, 1.0] if family == FAMILIES[0] else [0.0]):
        with pytest.raises(AnalysisError):
            fit(family, data.assign(y=value))


@pytest.mark.parametrize("family", FAMILIES)
def test_exported_original_unit_path_reconstructs_selected_prediction(family):
    data = frame(family)
    model = fit(family, data, lambda_path=[0.025, 0.3, 0.12], weights="w")
    path = oe.regularized_glm_path(model)
    assert list(path.columns) == ["Penalty", "Term", "Estimate", "Objective", "KKT", "Iterations", "Selected"]
    assert set(path.Penalty) == {0.3, 0.12, 0.025}
    assert path.attrs["inference"] == "unavailable"
    selected = path.loc[path.Selected]
    assert set(selected.Term) == {"Intercept", "a", "z"}
    values = selected.set_index("Term").Estimate
    link = values["Intercept"] + data.a * values["a"] + data.z * values["z"]
    np.testing.assert_allclose(link, prediction(model, data, kind="link"), rtol=1e-12, atol=1e-13)
    # The loss part is a weighted mean, and factors multiply both penalty
    # components in standardized coordinates. No likelihood df is invented.
    s = state(model)
    for point in s["paths"]:
        _, z = encoded(data, s["design"])
        beta = np.asarray(point["coefficients"])
        eta = z @ beta + point["constant"]
        penalty = point["penalty"] * np.dot(
            s["design"]["effective_factors"], 0.5 * np.abs(beta) + 0.25 * beta**2
        )
        expected = np.dot(data.w, observation_loss(eta, data.y.to_numpy(), family)) / data.w.sum() + penalty
        np.testing.assert_allclose(point["objective"], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_weight_category_is_not_learned_from_excluded_row(family):
    data = frame(family)
    data.iloc[0, data.columns.get_loc("w")] = 0
    data.iloc[0, data.columns.get_loc("g")] = "zero-weight-only"
    model = fit(family, data, x=["a", "g"], categorical=["g"], weights="w")
    assert {v["value"] for v in state(model)["design"]["levels"]["g"]} == {"low", "middle", "high"}
    with pytest.raises(AnalysisError):
        oe.regularized_glm_predict(model, data.iloc[[0]])


@pytest.mark.parametrize("family", FAMILIES)
def test_selected_prediction_digest_metadata_and_missing_row_readback(family):
    model = fit(family)
    query = frame(family).iloc[[3, 3, 17, 21]][["a", "z"]].copy()
    query.iloc[1, 0] = np.nan
    out = oe.regularized_glm_predict(model, query, missing="drop")
    assert out.attrs["row_positions"] == [0, 2, 3]
    assert out.attrs["dropped_rows"] == 1
    assert out.attrs["source_result_id"] == model.id
    assert out.attrs["state_digest"] == state(model)["digest"]
    assert out.attrs["inference"] == "unavailable"


def test_saved_poisson_overflow_cannot_produce_nonfinite_prediction():
    model = fit(FAMILIES[1])
    coefficient = state(model)["paths"][state(model)["selected_index"]]["coefficients"][0]
    assert coefficient != 0
    query = pd.DataFrame({"a": [np.copysign(1e100, coefficient)], "z": [0.0]})
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_glm_predict(model, query)
    assert caught.value.code == "numerical_failure"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("corruption", [
    "selected_bool", "center_length", "factor_bool", "penalty_mismatch",
    "coefficient_width", "coefficient_bool", "negative_kkt", "kkt_too_large",
    "converged_truthy", "iteration_bool", "raw_normalized_disagree",
    "normalized_nonunit", "duplicate_physical_positions", "wrong_family",
    "wrong_estimator", "wrong_sample", "raw_zero", "fixed_cv_data", "grid_reverse",
])
def test_checksum_recomputed_fixed_state_semantic_corruption_is_refused(family, corruption):
    result = fit(family, lambda_path=[0.3, 0.12, 0.025]).model_copy(deep=True)
    s = state(result)
    point = s["paths"][0]
    if corruption == "selected_bool":
        s["selected_index"] = True
    elif corruption == "center_length":
        s["design"]["centers"] = []
    elif corruption == "factor_bool":
        s["design"]["effective_factors"][0] = True
    elif corruption == "penalty_mismatch":
        point["penalty"] += 0.01
    elif corruption == "coefficient_width":
        point["coefficients"].append(0.0)
    elif corruption == "coefficient_bool":
        point["coefficients"][0] = True
    elif corruption == "negative_kkt":
        point["kkt_max"] = -1.0
    elif corruption == "kkt_too_large":
        point["kkt_max"] = 10 * point["kkt_limit"]
    elif corruption == "converged_truthy":
        point["converged"] = 1
    elif corruption == "iteration_bool":
        point["iterations"] = True
    elif corruption == "raw_normalized_disagree":
        s["raw_weights"][0] *= 2
    elif corruption == "normalized_nonunit":
        s["normalized_weights"][0] += 0.1
    elif corruption == "duplicate_physical_positions":
        s["physical_positions"][1] = s["physical_positions"][0]
        result.sample_positions = list(s["physical_positions"])
    elif corruption == "wrong_family":
        s["family"] = "poisson" if family == FAMILIES[0] else "binomial"
    elif corruption == "wrong_estimator":
        s["spec"]["estimator"] = FAMILIES[1] if family == FAMILIES[0] else FAMILIES[0]
    elif corruption == "wrong_sample":
        result.sample_positions = result.sample_positions[1:]
    elif corruption == "raw_zero":
        s["raw_weights"][0] = 0
    elif corruption == "fixed_cv_data":
        s["cv_scores"] = [1.0, 2.0, 3.0]
    elif corruption == "grid_reverse":
        s["lambda_path"].reverse()
    assert_helpers_refuse(resigned(result), family)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("corruption", [
    "training_overlap", "foreign_validation_row", "assignment_bool",
    "cv_grid_length", "cv_grid_mismatch", "cv_path_length",
    "cv_design_centers", "cv_loss_length", "cv_fold_length", "cv_scores_length",
])
def test_checksum_recomputed_cv_state_partition_and_schema_corruption_refused(family, corruption):
    result = fit(family, selection="cv", penalty=None,
                 lambda_path=[0.3, 0.1], folds=3).model_copy(deep=True)
    s = state(result)
    fold = s["cv"][0]
    if corruption == "training_overlap":
        fold["training_positions"].append(fold["validation_positions"][0])
    elif corruption == "foreign_validation_row":
        fold["validation_positions"][0] = 9999
    elif corruption == "assignment_bool":
        s["fold_assignments"][0] = False
    elif corruption == "cv_grid_length":
        fold["lambda_path"].pop()
    elif corruption == "cv_grid_mismatch":
        fold["lambda_path"][0] += 0.01
    elif corruption == "cv_path_length":
        fold["paths"].pop()
    elif corruption == "cv_design_centers":
        fold["design"]["centers"] = []
    elif corruption == "cv_loss_length":
        fold["validation_loss"].pop()
    elif corruption == "cv_fold_length":
        s["cv"].pop()
    elif corruption == "cv_scores_length":
        s["cv_scores"].pop()
    assert_helpers_refuse(resigned(result), family)


@pytest.mark.parametrize("family", FAMILIES)
def test_frequency_total_and_physical_nobs_are_separate_and_replay_bound(family):
    data = frame(family)
    result = fit(family, data, weights="w", weight_type="fweight")
    assert result.nobs == len(data)
    assert state(result)["physical_nobs"] == len(data)
    assert state(result)["frequency_total"] == int(data.w.sum())
    mutated = result.model_copy(deep=True)
    state(mutated)["frequency_total"] += 1
    assert_helpers_refuse(resigned(mutated), family)


def test_saved_poisson_response_underflow_is_refused_without_clipping():
    model = fit(FAMILIES[1])
    coefficient = state(model)["paths"][state(model)["selected_index"]]["coefficients"][0]
    assert coefficient != 0
    query = pd.DataFrame({"a": [np.copysign(1e100, -coefficient)], "z": [0.0]})
    eta = prediction(model, query, kind="link")
    assert np.isfinite(eta).all() and eta[0] < -700
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_glm_predict(model, query, kind="response")
    assert caught.value.code == "numerical_failure"


@pytest.mark.parametrize("family", FAMILIES)
def test_original_int64_frequency_count_above_float64_exact_domain_is_rejected(family):
    data = frame(family)
    data["w"] = np.ones(len(data), dtype=np.int64)
    data.iloc[0, data.columns.get_loc("w")] = 2**53 + 1
    original = data.copy(deep=True)
    assert int(data.w.iloc[0]) == 2**53 + 1
    with pytest.raises(AnalysisError) as caught:
        fit(family, data, weights="w", weight_type="fweight")
    assert caught.value.code == "invalid_weights"
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_frequency_upper_boundary_has_normalized_prediction_and_integer_total(family):
    data = frame(family)
    data["w"] = np.full(len(data), 2**53, dtype=np.int64)
    weighted = fit(family, data, weights="w", weight_type="fweight")
    unweighted = fit(family, data)
    assert state(weighted)["frequency_total"] == len(data) * 2**53
    assert weighted.nobs == len(data)
    np.testing.assert_allclose(prediction(weighted, data), prediction(unweighted, data), rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("kind", ["link", "response"])
def test_logistic_sigmoid_cannot_hide_overflowing_finite_query_link(kind):
    data = frame(FAMILIES[0])
    data["a"] = np.tile([-1.0, 1.0], len(data) // 2)
    data["y"] = (data.a > 0).astype(float)
    model = fit(FAMILIES[0], data, x=["a"], penalty=0.01, standardize=False)
    coefficient = state(model)["paths"][0]["coefficients"][0]
    assert abs(coefficient) > 1
    query = pd.DataFrame({"a": [np.copysign(np.finfo(np.float64).max, coefficient)]})
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_glm_predict(model, query, kind=kind)
    assert caught.value.code == "numerical_failure"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("corruption", ["factor_binding", "forced_binding", "standardize_binding",
                                         "physical_nobs", "dropped_rows", "original_range"])
def test_recomputed_checksum_cannot_rebind_declared_options_or_sample_counts(family, corruption):
    model = fit(family, standardize=False).model_copy(deep=True)
    s = state(model)
    if corruption == "factor_binding":
        s["design"]["penalty_factors"]["a"] = 2.0
        s["design"]["effective_factors"][0] = 2.0
    elif corruption == "forced_binding":
        s["design"]["penalty_factors"]["a"] = 0.0
        s["design"]["effective_factors"][0] = 0.0
        s["design"]["forced_controls"] = ["a"]
    elif corruption == "standardize_binding":
        s["design"]["standardize"] = True
    elif corruption == "physical_nobs":
        model.nobs += 1
    elif corruption == "dropped_rows":
        model.dropped_rows += 1
    elif corruption == "original_range":
        s["physical_positions"][-1] = model.nobs_original
        model.sample_positions = list(s["physical_positions"])
    assert_helpers_refuse(resigned(model), family)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("corruption", ["full_supplied_grid", "fold_supplied_grid", "pooled_loss",
                                         "loss_bool", "chosen_index", "fold_standardize"])
def test_recomputed_checksum_cannot_rebind_complete_supplied_cv_selector(family, corruption):
    model = fit(family, selection="cv", penalty=None,
                lambda_path=[0.3, 0.1], folds=3, standardize=False).model_copy(deep=True)
    s = state(model)
    fold = s["cv"][0]
    if corruption == "full_supplied_grid":
        s["lambda_path"][0] += 0.01
        s["paths"][0]["penalty"] = s["lambda_path"][0]
    elif corruption == "fold_supplied_grid":
        fold["lambda_path"][0] += 0.01
        fold["paths"][0]["penalty"] = fold["lambda_path"][0]
    elif corruption == "pooled_loss":
        s["cv_scores"] = [v + 1.0 for v in s["cv_scores"]]
    elif corruption == "loss_bool":
        fold["validation_loss"][0] = True
    elif corruption == "chosen_index":
        s["selected_index"] = 1 - s["selected_index"]
    elif corruption == "fold_standardize":
        fold["design"]["standardize"] = True
    assert_helpers_refuse(resigned(model), family)


@pytest.mark.parametrize("family", FAMILIES)
def test_auto_fraction_geometry_remains_bound_to_declared_selector(family):
    model = fit(family, selection="cv", penalty=None, n_lambdas=3,
                lambda_ratio=0.1, folds=3).model_copy(deep=True)
    s = state(model)
    s["fractions"][1] *= 1.1
    # Consistently change every grid and path penalty: their declared geometric
    # selector must still match the immutable specification.
    s["lambda_path"][1] = s["lambda_path"][0] * s["fractions"][1]
    s["paths"][1]["penalty"] = s["lambda_path"][1]
    for fold in s["cv"]:
        fold["lambda_path"][1] = fold["lambda_path"][0] * s["fractions"][1]
        fold["paths"][1]["penalty"] = fold["lambda_path"][1]
    assert_helpers_refuse(resigned(model), family)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("value", [-1.0, True, "0", float("inf"), float("nan"), 0.001])
def test_saved_free_direction_convergence_certificate_cannot_be_corrupted(family, value):
    model = fit(family).model_copy(deep=True)
    state(model)["paths"][0]["free_newton_correction"] = value
    # NaN/Inf cannot have a valid canonical digest. The extended JSON checksum
    # explicitly exercises their refusal without pretending it is admissible.
    nonfinite = isinstance(value, float) and not np.isfinite(value)
    assert_helpers_refuse(resigned(model, allow_nan=nonfinite), family)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("at_boundary", [False, True])
def test_saved_free_direction_certificate_accepts_zero_and_declared_boundary(family, at_boundary):
    model = fit(family)
    before = prediction(model, frame(family))
    restored = model.model_copy(deep=True)
    limit = np.sqrt(restored.spec.options["tolerance"])
    state(restored)["paths"][0]["free_newton_correction"] = float(limit) if at_boundary else 0.0
    resigned(restored)
    np.testing.assert_array_equal(prediction(restored, frame(family)), before)
    pd.testing.assert_frame_equal(oe.regularized_glm_path(restored), oe.regularized_glm_path(model), check_flags=False)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_saved_training_fold_free_direction_certificate_is_checked(family):
    model = fit(family, selection="cv", penalty=None,
                lambda_path=[0.3, 0.1], folds=3).model_copy(deep=True)
    state(model)["cv"][0]["paths"][1]["free_newton_correction"] = 1.0
    assert_helpers_refuse(resigned(model), family)
