"""Adversarial saved empirical-target state and scientific replay boundaries."""

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survey import replicate_margins_state as implementation
from openecon.resources import use_workspace_budget
from test_survey_replicate_margins_math import fixture, fit


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("brr", "fay", "jackknife", "bootstrap")


def rehash(payload):
    values = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    payload["integrity_sha256"] = hashlib.sha256(json.dumps(
        values, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return payload


@pytest.fixture(scope="module")
def complete_states():
    frame, design, _ = fixture()
    return {
        (method, target): fit(frame, design, "probit", method, target, alpha=.1,
                             null=[.1, .2, .3] if target == "mean" else [.1, -.2])
        for method in METHODS for target in ("mean", "ame")
    }


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target", ("mean", "ame"))
def test_sorted_json_roundtrip_keeps_physical_targets_and_full_t_inference(complete_states, method, target):
    state = complete_states[method, target]
    payload = state.to_state()
    restored = oe.SurveyReplicateMarginsResult.from_state(json.dumps(payload, sort_keys=True))
    assert restored.to_state() == payload
    assert restored.labels == state.labels
    assert restored.source_result.model_dump(mode="json") == payload["source_result"]
    pd.testing.assert_frame_equal(restored.to_frame(), state.to_frame())
    assert restored.to_frame().attrs == state.to_frame().attrs
    assert restored.null == ((.1, .2, .3) if target == "mean" else (.1, -.2))
    assert len(restored.null) != len(restored.source_result.null) if target == "ame" else True


def test_explicit_profile_labels_survive_non_alphabetical_native_json_encoding():
    frame, design, _ = fixture()
    state = fit(frame, design, "probit", "fay", "mean",
                profiles={"z-last": {"x": -.7}, "a-first": {"x": .4}, "middle": {"x": 1.2}})
    encoded = json.dumps(state.to_state(), sort_keys=True, ensure_ascii=False, allow_nan=False)
    decoded = json.loads(encoded)
    assert list(decoded["profiles"]) == ["a-first", "middle", "z-last"]
    assert decoded["labels"] == ["z-last", "a-first", "middle"]
    restored = oe.SurveyReplicateMarginsResult.from_state(encoded)
    assert restored.labels == state.labels
    assert restored.estimates == state.estimates and restored.covariance == state.covariance
    pd.testing.assert_frame_equal(restored.to_frame(), state.to_frame())


@pytest.mark.parametrize("target", ("mean", "ame"))
def test_state_and_frame_are_copy_on_read_without_original_data_or_new_fits(monkeypatch, complete_states, target):
    state = complete_states["fay", target]
    expected = state.to_state()

    def unexpected(*args, **kwargs):
        raise AssertionError("Saved targets attempted fitting or source-data admission.")

    monkeypatch.setattr("openecon.econometrics.survey.regression.fit_sample", unexpected)
    monkeypatch.setattr("openecon.econometrics.survey.regression_common.admit", unexpected)
    payload = state.to_state()
    restored = oe.SurveyReplicateMarginsResult.from_state(payload)
    payload["metadata"]["primitive"]["base_weights"][0] = -8.
    assert restored.to_state() == expected
    first = restored.to_frame()
    first.attrs["survey_replicate_margins_state"]["metadata"]["primitive"]["X"][0][0] = -12.
    first.attrs["covariance_matrix"][0][0] = -9.
    first.iloc[0, 0] = -10.
    assert restored.to_state() == expected
    pd.testing.assert_frame_equal(restored.to_frame(), state.to_frame())


def mutate(payload, defect):
    primitive = payload["metadata"]["primitive"]
    source = payload["source_result"]
    if defect == "estimate":
        payload["estimates"][0] += .01
    elif defect == "replica-estimate":
        payload["replicate_estimates"][1][0] += .02
    elif defect == "off-diagonal-covariance":
        payload["covariance"][0][1] += .001
        payload["covariance"][1][0] += .001
    elif defect == "diagonal-covariance":
        payload["covariance"][0][0] *= 1.1
    elif defect == "negative-variance":
        payload["covariance"][0][0] = -1.
    elif defect == "sample-X":
        primitive["X"][0][1] += .5
    elif defect == "sample-y":
        primitive["y"][0] = 1 - primitive["y"][0]
    elif defect == "intercept":
        primitive["X"][0][0] = .5
    elif defect == "physical-position":
        primitive["sample_positions"][0] = primitive["sample_positions"][1]
    elif defect == "whole-PSU-factor":
        primitive["psu_factors"][0][0] += .1
    elif defect == "actual-replica-weight":
        primitive["replicate_weights"][0][0] += .1
    elif defect == "complete-base-weight":
        primitive["base_weights"][0] *= 1.1
    elif defect == "complete-PSU-group":
        primitive["groups"][0] = primitive["groups"][1] + 1
    elif defect == "missing-full-weight-row":
        primitive["replicate_weights"][0].pop()
    elif defect == "missing-census-or-domain-PSU":
        primitive["psu_factors"][0].pop()
    elif defect == "missing-replica":
        payload["replicate_estimates"].pop()
    elif defect == "target-null-width":
        payload["null"].pop()
    elif defect == "label-order":
        payload["labels"] = payload["labels"][::-1]
    elif defect == "profile-value":
        profile = next(iter(payload["profiles"].values()))
        profile["x"] += .5
    elif defect == "complete-fixed-profile":
        next(iter(payload["profiles"].values()))["z"] = 0.
    elif defect == "uncertainty-claim":
        payload["metadata"]["empirical_covariate_uncertainty"] = False
    elif defect == "precision-claim":
        payload["metadata"]["precision"] = "float32"
    elif defect == "device-claim":
        payload["metadata"]["device"] = "cuda"
    elif defect == "actual-work-underreported":
        payload["metadata"]["work"]["actual_work_units"] = source["metadata"]["replication"]["work"]["actual_work_units"]
    elif defect == "replay-work-underreported":
        payload["metadata"]["work"]["target_replay_work_units"] -= 1
    elif defect == "one-byte-workspace":
        payload["metadata"]["workspace"]["estimated_workspace_bytes"] = 1
        payload["metadata"]["workspace"]["buffers"] = {"forged": 1}
    elif defect == "extra-primitive":
        primitive["hidden-covariate-filter"] = [True]
    elif defect == "saved-gradient":
        source["metadata"]["convergence"]["score_max_abs"] += 1.
        rehash(source)
    elif defect == "replica-gradient":
        source["metadata"]["replication"]["replicate_convergence"][0]["score_max_abs"] += 1.
        rehash(source)
    else:
        raise AssertionError(defect)


@pytest.mark.parametrize("defect", [
    "estimate", "replica-estimate", "off-diagonal-covariance", "diagonal-covariance",
    "negative-variance", "sample-X", "sample-y", "intercept", "physical-position",
    "whole-PSU-factor", "actual-replica-weight", "complete-base-weight", "complete-PSU-group",
    "missing-full-weight-row", "missing-census-or-domain-PSU", "missing-replica",
    "target-null-width", "label-order", "profile-value", "complete-fixed-profile",
    "uncertainty-claim", "precision-claim", "device-claim", "actual-work-underreported",
    "replay-work-underreported", "one-byte-workspace", "extra-primitive",
    "saved-gradient", "replica-gradient",
])
def test_recomputed_digest_does_not_admit_scientifically_corrupted_target(complete_states, defect):
    payload = complete_states["brr", "mean"].to_state()
    mutate(payload, defect)
    rehash(payload)
    with pytest.raises((ValueError, AnalysisError)):
        oe.SurveyReplicateMarginsResult.from_state(payload)


@pytest.mark.parametrize("path", ("X", "base_weights", "replicate_weights", "psu_factors", "estimates", "null"))
@pytest.mark.parametrize("value", (True, "1.0", complex(1, 2), float("nan"), float("inf")))
def test_raw_numeric_schema_never_coerces_complex_bool_strings_or_nonfinite(complete_states, path, value):
    payload = complete_states["fay", "ame"].to_state()
    if path in {"estimates", "null"}:
        payload[path][0] = value
    elif path in {"X", "replicate_weights", "psu_factors"}:
        payload["metadata"]["primitive"][path][0][0] = value
    else:
        payload["metadata"]["primitive"][path][0] = value
    with pytest.raises((ValueError, AnalysisError)):
        oe.SurveyReplicateMarginsResult.from_state(payload)


@pytest.mark.parametrize("value", ("64", True, -1, 513, 64.))
def test_raw_declared_memory_is_admitted_before_workspace_arithmetic(monkeypatch, complete_states, value):
    payload = complete_states["fay", "mean"].to_state()
    payload["source_result"]["design"]["max_memory_mb"] = value

    def unexpected(*args, **kwargs):
        raise AssertionError("Unadmitted memory schema reached workspace multiplication.")

    monkeypatch.setattr(implementation, "_workspace", unexpected)
    with pytest.raises(ValueError):
        oe.SurveyReplicateMarginsResult.from_state(payload)


def test_oversized_replay_fails_before_tensor_conversion_or_dictionary_copy(monkeypatch, complete_states):
    payload = complete_states["fay", "mean"].to_state()
    payload["source_result"]["design"]["validation"]["nobs"] = 4_000_000

    def unexpected(*args, **kwargs):
        raise AssertionError("Oversized saved replay allocated or copied before admission.")

    monkeypatch.setattr(implementation, "deepcopy", unexpected)
    monkeypatch.setattr(torch, "tensor", unexpected)
    with pytest.raises(AnalysisError) as error:
        oe.SurveyReplicateMarginsResult.from_state(payload)
    assert error.value.code == "survey_regression_budget"


def test_current_workspace_budget_fails_before_tensor_conversion(monkeypatch, complete_states):
    payload = complete_states["fay", "mean"].to_state()
    monkeypatch.setattr(implementation, "workspace_budget_bytes", lambda: 1)

    def unexpected(*args, **kwargs):
        raise AssertionError("Insufficient saved replay workspace reached tensor conversion.")

    monkeypatch.setattr(torch, "tensor", unexpected)
    with pytest.raises(AnalysisError) as error:
        oe.SurveyReplicateMarginsResult.from_state(payload)
    assert error.value.code == "workspace_limit"


def test_declared_memory_limit_also_bounds_restore_workspace(monkeypatch, complete_states):
    payload = complete_states["fay", "mean"].to_state()
    payload["source_result"]["design"]["max_memory_mb"] = 1
    payload["source_result"]["design"]["validation"]["nobs"] = 20_000

    def unexpected(*args, **kwargs):
        raise AssertionError("Declared workspace excess reached tensor allocation.")

    monkeypatch.setattr(torch, "tensor", unexpected)
    with use_workspace_budget(512), pytest.raises(AnalysisError) as error:
        oe.SurveyReplicateMarginsResult.from_state(payload)
    assert error.value.code == "workspace_limit"
    assert error.value.resource_plan["budget_bytes"] == 1024**2


def _covariance(original, replicas, record):
    values = np.asarray(replicas)
    if record["centering"] == "original":
        difference = values - np.asarray(original)
    elif record["centering"] == "replicate_mean":
        difference = values - values.mean(0)
    else:
        difference = np.empty_like(values)
        strata = np.asarray(record["replicate_stratum_indices"])
        for h in np.unique(strata):
            difference[strata == h] = values[strata == h] - values[strata == h].mean(0)
    return (difference.T * record["variance_multipliers"]) @ difference


def test_coherent_beta_target_and_covariance_rehash_still_needs_actual_likelihood_solution(complete_states):
    payload = complete_states["brr", "mean"].to_state()
    source = payload["source_result"]
    source["coefficients"][0] += .1
    record = source["metadata"]["replication"]
    source["covariance"] = _covariance(source["coefficients"], record["replicate_estimates"], record).tolist()
    rehash(source)
    # The old coefficient envelope has no original observations. This attack
    # is internally coherent there; the new primitive binding adds the guard.
    oe.SurveyRegressionResult.model_validate(source)
    primitive = payload["metadata"]["primitive"]
    ordered = {label: payload["profiles"][label] for label in payload["labels"]}
    payload["estimates"] = implementation.evaluate_targets(
        torch.tensor(primitive["X"], dtype=torch.float64),
        torch.tensor(source["coefficients"], dtype=torch.float64),
        torch.tensor(primitive["base_weights"], dtype=torch.float64), primitive["sample_positions"],
        family=source["family"], labels=source["labels"], target=payload["target"], profiles=ordered,
        variables=payload["variables"], regressors=source["regressors"], intercept=source["intercept"],
    ).tolist()
    payload["covariance"] = _covariance(payload["estimates"], payload["replicate_estimates"], record).tolist()
    rehash(payload)
    with pytest.raises(ValueError, match="weighted score equations"):
        oe.SurveyReplicateMarginsResult.from_state(payload)


@pytest.mark.parametrize("mode", ("default", "float32", "meta", "inference"))
def test_saved_target_replay_is_cpu_float64_under_ambient_torch_defaults(complete_states, mode):
    payload = complete_states["jackknife", "ame"].to_state()
    expected = complete_states["jackknife", "ame"].to_frame()
    original = torch.get_default_dtype()
    try:
        if mode == "float32":
            torch.set_default_dtype(torch.float32)
        context = torch.device("meta") if mode == "meta" else torch.inference_mode() if mode == "inference" else nullcontext()
        with context:
            restored = oe.SurveyReplicateMarginsResult.from_state(payload)
            frame = restored.to_frame()
        assert restored.to_state() == payload
        pd.testing.assert_frame_equal(frame, expected)
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("family,index,expected", [
    ("probit", -10., math.erfc(10 / math.sqrt(2)) / 2),
    ("logit", -710., math.exp(-710)),
])
def test_representable_negative_tail_targets_are_not_rounded_to_zero(family, index, expected):
    X = torch.tensor([[1., 0., 0.], [1., 1., 1.]], dtype=torch.float64)
    beta = torch.tensor([index, 0., 0.], dtype=torch.float64)
    actual = implementation.evaluate_targets(
        X, beta, torch.ones(2, dtype=torch.float64), [0, 1], family=family,
        labels=["_cons", "x", "z"], target="mean", profiles={"partial": {"x": 0.}},
        variables=(), regressors=["x", "z"], intercept=True,
    )
    assert actual[0] > 0
    assert math.isclose(float(actual[0]), expected, rel_tol=1e-12, abs_tol=0.)


def test_positive_target_weight_underflow_is_not_silently_omitted():
    with pytest.raises(AnalysisError, match="weight underflowed"):
        implementation.evaluate_targets(
            torch.tensor([[1., 0.], [1., 1.]], dtype=torch.float64),
            torch.tensor([0., .2], dtype=torch.float64),
            torch.tensor([1e-320, 1e308], dtype=torch.float64), [0, 1], family="probit",
            labels=["_cons", "x"], target="ame", profiles={}, variables=["x"],
            regressors=["x"], intercept=True,
        )


def test_tiny_finite_target_cannot_be_replaced_by_zero_with_a_new_digest():
    frame, design, _ = fixture()
    result = fit(frame, design, "probit", "fay", "mean", profiles={"tiny": {"x": 150.}})
    payload = result.to_state()
    assert 0 < payload["estimates"][0] < 1e-40
    assert oe.SurveyReplicateMarginsResult.from_state(json.dumps(payload, sort_keys=True)).to_state() == payload
    payload["estimates"][0] = 0.
    rehash(payload)
    with pytest.raises(ValueError, match="empirical targets"):
        oe.SurveyReplicateMarginsResult.from_state(payload)


def test_positive_subnormal_covariance_cannot_be_replaced_by_zero_with_a_new_digest():
    frame, design, _ = fixture()
    result = fit(frame, design, "logit", "fay", "mean", profiles={"tiny": {"x": 4725.}})
    payload = result.to_state()
    variance = payload["covariance"][0][0]
    assert 0 < variance < 1e-310
    assert result.to_frame().loc["tiny", "std_error"] > 0
    assert oe.SurveyReplicateMarginsResult.from_state(payload).covariance == result.covariance
    payload["covariance"][0][0] = 0.
    rehash(payload)
    with pytest.raises(ValueError, match="covariance"):
        oe.SurveyReplicateMarginsResult.from_state(payload)


@pytest.mark.parametrize("where", ("original", "replica"))
def test_poisson_cancellation_bound_still_rejects_forged_gradient_diagnostics(where):
    frame, design, _ = fixture()
    x, z = frame.x.tolist(), frame.z.tolist()
    frame["c"] = [round(1e12 * math.exp(.2 * a - .1 * b)) for a, b in zip(x, z)]
    frame["x"] *= 2.7
    frame["z"] /= 5.
    result = fit(frame, design, "poisson", "fay", "mean",
                 profiles={"low": {"x": -.8 * 2.7}, "high": {"x": 1.2 * 2.7}})
    payload = result.to_state()
    source = payload["source_result"]
    evidence = (source["metadata"]["convergence"] if where == "original"
                else source["metadata"]["replication"]["replicate_convergence"][0])
    evidence["score_max_abs"] += 1000.
    rehash(source)
    rehash(payload)
    with pytest.raises(ValueError, match="convergence differs"):
        oe.SurveyReplicateMarginsResult.from_state(payload)


@pytest.mark.parametrize("path", [
    "docs/evidence/survey-regression-2026-10-07/persisted-states.json",
    "docs/evidence/survey-regression-replication-2026-10-07/persisted-states.json",
    "docs/evidence/survey-probit-poisson-replication-2026-10-07/persisted-states.json",
])
def test_prior_taylor_and_coefficient_replication_states_keep_identical_payloads(path):
    payloads = json.loads((ROOT / path).read_text())
    assert len(payloads) >= 4
    for payload in payloads.values():
        restored = oe.SurveyRegressionResult.model_validate_json(json.dumps(payload))
        assert restored.model_dump(mode="json") == payload
        assert restored.to_frame().attrs["survey_regression_state"] == payload
        with pytest.raises(ValueError):
            oe.SurveyReplicateMarginsResult.from_state(payload)
