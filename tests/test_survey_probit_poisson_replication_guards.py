"""Admission, complete replica failures, and saved-state guards for the new families.

Numerical coefficient/covariance oracles live in the separate mathematics tests.
These fixtures deliberately retain a valid original model while removing its
identifying or finite-fit support in specific whole-PSU replicas.
"""

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget
from test_survey_regression_replication import fixture as previous_fixture


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("probit", "poisson")
METHODS = ("brr", "fay", "jackknife", "bootstrap")


def fixture():
    frame, design, bootstrap = previous_fixture()
    frame["c"] = np.tile([0, 1, 2, 0, 3, 1, 0, 2], 6)
    return frame, design, bootstrap


def fit(frame, design, family, method, *, supplied=None, **overrides):
    options = {"tolerance": 1e-11}
    if method in {"brr", "fay"}:
        options["replicates"] = 4
    if method == "fay":
        options["rho"] = .5
    if method == "bootstrap":
        options.update(
            replicate_weights=supplied, scale=1/len(supplied.columns),
            justification="Declared whole-PSU test perturbations; no row resampling.",
        )
    options.update(overrides)
    function = getattr(oe, f"survey_{family}_replicate")
    return function(frame, design, "b" if family == "probit" else "c", ["x", "z"],
                    method=method, **options)


@pytest.fixture(scope="module")
def complete_states():
    frame, design, bootstrap = fixture()
    return {
        f"{family}_{method}": fit(frame, design, family, method,
                                supplied=bootstrap if method == "bootstrap" else None)
        for family in FAMILIES for method in METHODS
    }


def rehash(payload):
    record = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    payload["integrity_sha256"] = hashlib.sha256(json.dumps(
        record, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return payload


def failure_fixture(family, *, rank=False):
    frame = pd.DataFrame({
        "h": np.repeat(range(3), 8), "p": np.tile(np.repeat([0, 1], 4), 3),
        "w": 1., "x": np.tile([-2., -1., 1., 2.], 6),
        "z": np.tile([.4, -.3, .5, -.7], 6),
    })
    if rank:
        # Every repeated covariate has mixed binary/count responses, so the
        # original finite-fit question does not depend on separation heuristics.
        frame["x"] = np.tile([-1., -1., .7, .7], 6)
        frame["z"] = np.tile([.4, .4, -.3, -.3], 6)
        frame.loc[4:, "z"] = 0.
        frame["b"] = np.tile([0, 1], 12)
        frame["c"] = np.tile([0, 2], 12)
    else:
        frame["b"] = (frame.x > 0).astype(int)
        frame.loc[:3, "b"] = 1-frame.loc[:3, "b"]
        # The only positive-count PSU has a full-rank three-column design.
        # Removing it leaves an all-zero intercept model with no finite fit.
        frame["c"] = 0
        frame.loc[:3, "c"] = [2, 1, 3, 1]
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    original = getattr(oe, f"survey_{family}")(
        frame, design, "b" if family == "probit" else "c", ["x", "z"], tolerance=1e-11,
    )
    assert original.metadata["convergence"]["converged"] is True
    assert np.isfinite(original.coefficients).all()
    return frame, design


def support_removal_weights(frame):
    factor = np.ones(len(frame))
    factor[(frame.h == 0) & (frame.p == 0)] = 0.
    factor[(frame.h == 0) & (frame.p == 1)] = 2.
    return pd.DataFrame({
        "valid-before": frame.w.to_numpy(),
        "remove-support-a": frame.w.to_numpy()*factor,
        "valid-between": frame.w.to_numpy(),
        "remove-support-b": frame.w.to_numpy()*factor,
        "valid-after": frame.w.to_numpy(),
    })


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", ["brr", "jackknife", "bootstrap"])
@pytest.mark.parametrize("rank", [False, True], ids=["finite-fit-support", "rank-support"])
def test_valid_original_reports_all_bad_replica_ids_in_order(family, method, rank):
    frame, design = failure_fixture(family, rank=rank)
    original = frame.copy(deep=True)
    weights = support_removal_weights(frame) if method == "bootstrap" else None
    expected_ids = {
        "brr": ["sylvester-1", "sylvester-3"],
        "jackknife": ["stratum-0-delete-psu-0"],
        "bootstrap": ["remove-support-a", "remove-support-b"],
    }[method]
    with pytest.raises(AnalysisError) as error:
        fit(frame, design, family, method, supplied=weights)
    assert error.value.code == "survey_replicate_failure"
    failures = error.value.failed_replicates
    assert [item["replicate_id"] for item in failures] == expected_ids
    expected_code = "survey_regression_rank" if rank else "survey_regression_separation"
    assert all(item["code"] == expected_code and item["n_used"] > 3 for item in failures)
    assert all(identity in str(error.value) for identity in expected_ids)
    assert 0 < error.value.work_record["actual_work_units"] <= 50_000_000
    # Later scientific replicas were attempted, rather than silently omitted.
    assert not hasattr(error.value, "unattempted_replicate_ids")
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("family", FAMILIES)
def test_positive_fay_retains_support_that_brr_cannot_fit(family):
    frame, design = failure_fixture(family)
    state = fit(frame, design, family, "fay", rho=.5)
    record = state.metadata["replication"]
    assert record["failed_replicates"] == []
    assert len(record["replicate_convergence"]) == 4
    assert all(item["n_used"] == len(frame) for item in record["replicate_convergence"])
    assert all(item["sample_positions"] == list(range(len(frame)))
               for item in record["replicate_convergence"])


def test_converged_original_does_not_hide_all_nonconverged_probit_replica_ids():
    frame, design, _ = fixture()
    original = oe.survey_probit(frame, design, "b", ["x", "z"], max_iter=3, tolerance=1e-11)
    assert original.metadata["convergence"]["converged"] is True
    with pytest.raises(AnalysisError) as error:
        fit(frame, design, "probit", "brr", max_iter=3)
    assert error.value.code == "survey_replicate_failure"
    assert [item["replicate_id"] for item in error.value.failed_replicates] == [
        "sylvester-0", "sylvester-3",
    ]
    assert all(item["code"] == "survey_regression_nonconvergence"
               for item in error.value.failed_replicates)


@pytest.mark.parametrize("value", [-1, .5, 2**53+1])
def test_poisson_raw_invalid_counts_are_not_repaired_by_float64_rounding(value):
    frame, design, _ = fixture()
    frame["c"] = frame.c.astype(object)
    frame.at[0, "c"] = value
    assert frame.at[0, "c"] == value
    with pytest.raises(AnalysisError) as error:
        fit(frame, design, "poisson", "fay")
    assert error.value.code == "invalid_survey_response"


@pytest.mark.parametrize("family,value", [("probit", .25), ("probit", 2),
                                         ("poisson", complex(1, 1))])
def test_nonbinary_or_complex_responses_have_no_implicit_conversion(family, value):
    frame, design, _ = fixture()
    outcome = "b" if family == "probit" else "c"
    frame[outcome] = frame[outcome].astype(object)
    frame.at[0, outcome] = value
    with pytest.raises(AnalysisError):
        fit(frame, design, family, "brr")


@pytest.mark.parametrize("family", FAMILIES)
def test_cumulative_work_is_refused_before_any_replica_weight_plan(monkeypatch, family):
    from openecon.econometrics.survey import regression_replication as implementation

    frame, design, _ = fixture()

    def unexpected(*args, **kwargs):
        raise AssertionError("Oversized complete fits reached replica-weight allocation.")

    monkeypatch.setattr(implementation, "plan_brr", unexpected)
    with pytest.raises(AnalysisError) as error:
        fit(frame, design, family, "brr", replicates=4096, max_iter=1000)
    assert error.value.code == "survey_regression_budget"


@pytest.mark.parametrize("family", FAMILIES)
def test_complete_live_workspace_is_admitted_before_replica_plan(monkeypatch, family):
    from openecon.econometrics.survey import regression_replication as implementation

    frame, design, _ = fixture()
    outcome = "b" if family == "probit" else "c"
    admitted = implementation.admit(frame, design, outcome, [])

    def unexpected(*args, **kwargs):
        raise AssertionError("Replica weights were allocated before complete workspace admission.")

    # Admit the small base sample first: this tests the cumulative replication
    # workspace, rather than merely triggering the earlier base-data guard.
    monkeypatch.setattr(implementation, "admit", lambda *args, **kwargs: admitted)
    monkeypatch.setattr(implementation, "plan_brr", unexpected)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        getattr(oe, f"survey_{family}_replicate")(
            frame, design, outcome, [], method="brr", replicates=128, max_iter=5,
        )
    assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("inference_mode", [False, True])
def test_cpu_float64_fit_restore_and_helpers_ignore_ambient_defaults(
    complete_states, family, inference_mode,
):
    frame, design, _ = fixture()
    baseline = complete_states[f"{family}_fay"]
    evaluation = frame[["x", "z"]].iloc[:3]
    expected_table = baseline.to_frame()
    expected_prediction = oe.survey_predict(baseline, evaluation)
    previous_device, previous_dtype = torch.get_default_device(), torch.get_default_dtype()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float32)
        with torch.inference_mode() if inference_mode else nullcontext():
            actual = fit(frame, design, family, "fay")
            assert actual == complete_states[f"{family}_fay"]
            restored = oe.SurveyRegressionResult.model_validate_json(actual.model_dump_json())
            assert restored.metadata["device"] == "cpu"
            assert restored.metadata["precision"] == "float64"
            pd.testing.assert_frame_equal(restored.to_frame(), expected_table)
            pd.testing.assert_frame_equal(oe.survey_predict(restored, evaluation), expected_prediction)
        assert torch.get_default_device().type == "meta"
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)


CORRUPTIONS = [
    ("duplicate_ids", "bootstrap"), ("wrong_convergence_id", "brr"),
    ("failed_replica", "jackknife"), ("not_converged", "fay"),
    ("short_support", "brr"), ("outside_support", "jackknife"),
    ("boolean_replica", "fay"), ("changed_replica", "bootstrap"),
    ("negative_variance", "brr"), ("asymmetric_variance", "fay"),
    ("covariance_replay", "bootstrap"), ("wrong_multiplier", "fay"),
    ("fay_rho", "fay"), ("wrong_jk_stratum", "jackknife"),
    ("bootstrap_scale", "bootstrap"), ("df_count", "bootstrap"),
    ("default_df", "brr"), ("cumulative_work", "fay"),
    ("missing_signs", "brr"), ("missing_fingerprint", "bootstrap"),
]


def corrupt(payload, kind):
    record = payload["metadata"]["replication"]
    convergence = record["replicate_convergence"][0]
    if kind == "duplicate_ids":
        record["replicate_ids"][1] = record["replicate_ids"][0]
    elif kind == "wrong_convergence_id":
        convergence["replicate_id"] = record["replicate_ids"][1]
    elif kind == "failed_replica":
        record["failed_replicates"] = [{"replicate_id": record["replicate_ids"][0]}]
    elif kind == "not_converged":
        convergence["converged"] = False
    elif kind == "short_support":
        convergence["sample_positions"].pop()
    elif kind == "outside_support":
        convergence["sample_positions"][-1] = payload["metadata"]["n_design"]
    elif kind == "boolean_replica":
        record["replicate_estimates"][0][0] = True
    elif kind == "changed_replica":
        record["replicate_estimates"][0][0] += .25
    elif kind == "negative_variance":
        payload["covariance"][0][0] = -1.
    elif kind == "asymmetric_variance":
        payload["covariance"][0][1] += 1.
    elif kind == "covariance_replay":
        payload["covariance"][0][0] += 1.
    elif kind == "wrong_multiplier":
        record["variance_multipliers"][0] *= 2
    elif kind == "fay_rho":
        record["fay_rho"] = 1.
    elif kind == "wrong_jk_stratum":
        record["replicate_stratum_indices"][-1] = 0
    elif kind == "bootstrap_scale":
        record["scale"] = 0.
    elif kind == "df_count":
        record["df_explicit"] = True
        payload["df"] = record["replicate_count"]
    elif kind == "default_df":
        payload["df"] = 1
    elif kind == "cumulative_work":
        convergence["work_units"] = 50_000_000
    elif kind == "missing_signs":
        record.pop("balanced_signs")
    elif kind == "missing_fingerprint":
        record.pop("replicate_weight_sha256")
    else:
        raise AssertionError(kind)
    return rehash(payload)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind,method", CORRUPTIONS)
def test_new_family_saved_contract_rejects_tampering_even_with_recomputed_digest(
    complete_states, family, kind, method,
):
    original = complete_states[f"{family}_{method}"].model_dump(mode="json")
    with pytest.raises(ValueError):
        oe.SurveyRegressionResult.model_validate(corrupt(deepcopy(original), kind))
    assert complete_states[f"{family}_{method}"].model_dump(mode="json") == original


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("helper", ["predict", "margins", "lincom", "test"])
def test_every_saved_helper_revalidates_mutated_typed_replica_state(complete_states, family, helper):
    state = complete_states[f"{family}_bootstrap"]
    metadata = deepcopy(state.metadata)
    metadata["replication"]["replicate_estimates"][0][0] += .25
    forged = state.model_copy(update={"metadata": metadata})
    frame, _, _ = fixture()
    functions = {
        "predict": lambda: oe.survey_predict(forged, frame[["x", "z"]].iloc[:3]),
        "margins": lambda: oe.survey_margins(forged, frame, variables=["x"], weights="w"),
        "lincom": lambda: oe.survey_lincom(forged, [1., .25, -.2]),
        "test": lambda: oe.survey_test(forged, [[0., 1., 0.], [0., 0., 1.]]),
    }
    with pytest.raises(ValueError):
        functions[helper]()
    assert state.metadata != metadata


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("guard", ["count", "source_cells", "replay_work", "workspace"])
def test_saved_new_family_replica_limits_precede_tensor_conversion(
    complete_states, monkeypatch, family, guard,
):
    from openecon.econometrics.survey import regression_common as implementation

    state = complete_states[f"{family}_bootstrap"]
    record = deepcopy(state.metadata["replication"])
    design = state.design
    if guard == "count":
        record["replicate_count"] = 4097
    elif guard == "source_cells":
        validation = design.validation.model_copy(update={"nobs": 8_000_001})
        design = design.model_copy(update={"validation": validation})
    elif guard == "replay_work":
        monkeypatch.setattr(implementation, "MAX_WORK", 1)
    else:
        monkeypatch.setattr(implementation, "workspace_budget_bytes", lambda: 1)

    def unexpected(*args, **kwargs):
        raise AssertionError("Saved replicas reached tensor conversion before bounded admission.")

    monkeypatch.setattr(implementation.torch, "tensor", unexpected)
    with pytest.raises((ValueError, AnalysisError)):
        implementation.covariance_from_replicates(
            state.coefficients, record, design, state.metadata["sample_positions"],
            original_work=state.metadata["convergence"]["work_units"],
        )


LEGACY_CASES = [("survey-regression-2026-10-07", family) for family in
                ("linear", "logit", "probit", "poisson")]
LEGACY_CASES += [("survey-regression-replication-2026-10-07", f"{family}_{method}")
                 for family in ("linear", "logit") for method in METHODS]


@pytest.mark.parametrize("directory,case", LEGACY_CASES)
def test_actual_pre_wave_native_states_keep_exact_logical_state_digest_and_helpers(directory, case):
    saved = json.loads((ROOT / f"docs/evidence/{directory}/persisted-states.json").read_text())[case]
    state = oe.SurveyRegressionResult.model_validate_json(json.dumps(saved, allow_nan=False))
    assert state.model_dump(mode="json") == saved
    assert state.integrity_sha256 == saved["integrity_sha256"]
    assert oe.SurveyRegressionResult.model_validate_json(state.model_dump_json()) == state
    table = state.to_frame()
    assert table.attrs["survey_regression_state"] == saved
    assert table.attrs["covariance_matrix"] == saved["covariance"]
    assert state.metadata["precision"] == "float64" and state.metadata["device"] == "cpu"
    kind = saved["metadata"].get("replication", {}).get("method", "Taylor")
    assert table.attrs["method"] == f"single-stage {kind} coefficient delta"
    evaluation = pd.DataFrame({column: [0., .25] for column in state.regressors},
                              index=["same", "same"])
    predicted = oe.survey_predict(state, evaluation)
    assert predicted.index.tolist() == ["same", "same"]
    assert predicted.attrs["physical_positions"] == [0, 1]
    assert predicted.attrs["survey_regression_state"] == saved
    assert oe.to_latex(table) and oe.to_latex(predicted)
