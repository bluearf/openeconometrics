"""SEM semantic replay, identification, dimensional and resident-input gates."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.latent.sem import (
    SEMState,
    _digest,
    latent_scores,
    latent_sem,
    latent_sem_covariance,
    latent_sem_restore,
    sem_effects,
)
from openecon.resources import use_workspace_budget

COLS = ["x1", "x2", "x3", "y1", "y2", "y3"]
FACTORS = {"z": COLS[:3], "a": COLS[3:]}
B = np.zeros((8, 8))
B[0, 6], B[1, 6], B[2, 6] = 1, 0.8, 0.7
B[3, 7], B[4, 7], B[5, 7] = 1, 0.9, 0.6
B[7, 6] = 0.5
A = np.linalg.inv(np.eye(8) - B)
OMEGA = A @ np.diag([0.4, 0.5, 0.6, 0.3, 0.4, 0.5, 1, 0.8]) @ A.T
PATHS = {"a": {"z": None}}


def summary(s=OMEGA[:6, :6], **options):
    return latent_sem_covariance(
        s.tolist(),
        columns=COLS,
        n=303,
        divisor="n",
        means=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        factors=FACTORS,
        paths=PATHS,
        tolerance=1e-8,
        **options,
    )


@pytest.fixture(scope="module")
def fitted():
    return summary()


@pytest.fixture(scope="module")
def sample():
    values = np.random.default_rng(362).multivariate_normal(np.arange(6) * 0.2, OMEGA[:6, :6], 127)
    frame = pd.DataFrame(
        values,
        columns=COLS,
        index=pd.MultiIndex.from_arrays(
            [np.arange(127), np.arange(127) % 5], names=["position", "group"]
        ),
    )
    frame.iloc[3, 2] = np.nan
    frame["x1"] = frame["x1"].astype("Float64")
    return frame, latent_sem(
        frame, columns=COLS, factors=FACTORS, paths=PATHS, missing="drop", tolerance=1e-8
    )


def rehash(state):
    state["digest"] = _digest({k: v for k, v in state.items() if k != "digest"})
    return state


def reject_state(state):
    with pytest.raises((AnalysisError, ValidationError, ValueError, TypeError)):
        latent_sem_restore(rehash(state))


def test_json_typed_and_table_state_roundtrip(fitted):
    payload = fitted.attrs["sem_state"]
    model = SEMState(payload=payload)
    for saved in (
        model,
        model.model_dump(),
        model.model_dump_json(),
        payload,
        json.dumps(payload, sort_keys=True),
        fitted,
    ):
        restored = latent_sem_restore(saved)
        pd.testing.assert_frame_equal(restored["parameters"], fitted["parameters"])
        pd.testing.assert_frame_equal(restored["covariance"], fitted["covariance"])
        assert restored.attrs["sem_state"]["digest"] == payload["digest"]
    assert isinstance(model.payload["solver_parameters"], tuple)
    with pytest.raises(TypeError):
        model.payload["spec"]["columns"][0] = "other"


def test_original_nullable_dtype_index_and_missing_sample_binding(sample):
    frame, result = sample
    state = result.attrs["sem_state"]
    assert state["source"]["dtypes"][0] == "Float64"
    assert state["source"]["positions"] == [i for i in range(len(frame)) if i != 3]
    before = latent_scores(result)
    after = latent_scores(latent_sem_restore(json.dumps(state)))
    assert before.index.identical(frame.index)
    assert before.iloc[3].isna().all()
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize(
    "field",
    [
        "covariance",
        "standardized_covariance",
        "natural_jacobian",
        "observed_information_solver",
        "implied_covariance",
        "paths",
        "disturbance_covariance",
        "node_covariance",
    ],
)
def test_rehashed_covariance_and_matrix_tampering_refuses(fitted, field):
    state = copy.deepcopy(fitted.attrs["sem_state"])
    state["results"][field][0][0] += 0.01
    reject_state(state)


@pytest.mark.parametrize("field", ["parameters", "intercepts", "node_means", "scales"])
def test_rehashed_parameter_and_mean_tampering_refuses(fitted, field):
    state = copy.deepcopy(fitted.attrs["sem_state"])
    state["results"][field][0] += 0.01
    reject_state(state)


@pytest.mark.parametrize(
    "field", ["statistic", "df", "baseline_statistic", "baseline_df", "CFI", "RMSEA", "SRMR"]
)
def test_rehashed_fit_diagnostic_tampering_refuses(fitted, field):
    state = copy.deepcopy(fitted.attrs["sem_state"])
    state["results"]["diagnostics"][field] = 91.0
    reject_state(state)


def test_rehashed_source_sample_dtype_and_constraints_tampering_refuses(sample):
    _, result = sample
    original = result.attrs["sem_state"]
    changes = [
        lambda s: s["source"]["positions"].append(3),
        lambda s: s["source"]["dtypes"].__setitem__(0, "int64"),
        lambda s: s["source"]["rows"][0].__setitem__(0, True),
        lambda s: s["spec"]["records"][0].__setitem__("fixed", 0.8),
        lambda s: s["starts"][0].__setitem__("start", False),
        lambda s: s["source"].__setitem__("original_n", True),
    ]
    for change in changes:
        state = copy.deepcopy(original)
        change(state)
        reject_state(state)


def test_tiny_original_unit_covariance_rehash_hole_is_refused():
    scale = np.array([1e-8, 1e-8, 1e-8, 2e-8, 2e-8, 2e-8])
    result = summary(OMEGA[:6, :6] * np.outer(scale, scale))
    original = result.attrs["sem_state"]
    names = original["results"]["parameter_names"]
    j = names.index("variance:x1")
    assert original["results"]["covariance"][j][j] < 1e-30
    state = copy.deepcopy(original)
    state["results"]["covariance"][j][j] = 1e-10
    reject_state(state)


def test_summary_coordinate_rescaling_preserves_paths_and_oim(fitted):
    scale = np.array([1e-7, 2e-7, 3e-7, 1e5, 2e5, 3e5])
    covariance = OMEGA[:6, :6] * np.outer(scale, scale)
    scaled = latent_sem_covariance(
        covariance.tolist(),
        columns=COLS,
        n=303,
        divisor="n",
        means=(np.arange(1, 7) * 0.1 * scale).tolist(),
        factors=FACTORS,
        paths=PATHS,
        tolerance=1e-8,
    )
    before = fitted.attrs["sem_state"]["results"]
    after = scaled.attrs["sem_state"]["results"]
    np.testing.assert_allclose(
        after["standardized_paths"], before["standardized_paths"], rtol=3e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        after["standardized_covariance"], before["standardized_covariance"], rtol=5e-5, atol=1e-9
    )
    units = np.r_[scale, scale[[0, 3]]]
    natural_units = []
    for name in before["parameter_names"]:
        family, expression = name.split(":", 1)
        if family in ("loading", "path"):
            target, source = expression.split("<-")
            natural_units.append(
                units[(COLS + list(FACTORS)).index(target)]
                / units[(COLS + list(FACTORS)).index(source)]
            )
        elif family == "variance":
            natural_units.append(units[(COLS + list(FACTORS)).index(expression)] ** 2)
        else:
            natural_units.append(units[(COLS + list(FACTORS)).index(expression)])
    np.testing.assert_allclose(
        after["covariance"],
        np.array(before["covariance"]) * np.outer(natural_units, natural_units),
        rtol=5e-5,
        atol=1e-28,
    )
    latent_sem_restore(json.dumps(scaled.attrs["sem_state"]))


def test_workspace_and_work_admission_precedes_input_copy(monkeypatch):
    frame = pd.DataFrame(np.ones((10, 6)), columns=COLS)
    called = []
    original = pd.DataFrame.copy

    def copied(self, *args, **kwargs):
        called.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", copied)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError, match="workspace"):
            latent_sem(frame, columns=COLS, factors=FACTORS, paths=PATHS)
    with pytest.raises(AnalysisError, match="work"):
        latent_sem(frame, columns=COLS, factors=FACTORS, paths=PATHS, max_work=1)
    assert called == []


def test_restore_work_admission_precedes_numerical_replay(fitted, sample, monkeypatch):
    import openecon.econometrics.latent.sem as module

    monkeypatch.setattr(
        module, "_evaluate", lambda *a, **k: pytest.fail("replay preceded admission")
    )
    state = copy.deepcopy(fitted.attrs["sem_state"])
    state["options"]["max_work"] = 1
    reject_state(state)
    with use_workspace_budget(1):
        with pytest.raises((AnalysisError, ValidationError), match="workspace"):
            latent_sem_restore(sample[1])


@pytest.mark.parametrize("construction", ["copy", "construct"])
@pytest.mark.parametrize("reader", ["restore", "tables", "effects", "scores", "dump", "json"])
def test_pydantic_validation_bypass_cannot_override_public_inference(fitted, construction, reader):
    original = fitted.attrs["sem_state"]
    forged = copy.deepcopy(original)
    forged["results"]["observed_information_solver"] = [
        [4 * v for v in row] for row in forged["results"]["observed_information_solver"]
    ]
    rehash(forged)
    model = (
        SEMState(payload=original).model_copy(update={"payload": forged})
        if construction == "copy"
        else SEMState.model_construct(payload=forged)
    )
    with pytest.raises((AnalysisError, PydanticSerializationError)):
        if reader == "restore":
            latent_sem_restore(model)
        elif reader == "tables":
            model.to_tables()
        elif reader == "effects":
            sem_effects(model, source="z", target="y2")
        elif reader == "dump":
            model.model_dump()
        elif reader == "json":
            model.model_dump_json()
        else:
            latent_scores(model, pd.DataFrame([[0.0] * 6], columns=COLS))


def test_typed_state_admission_precedes_serializer_thaw(fitted, monkeypatch):
    model = SEMState(payload=fitted.attrs["sem_state"])
    forged = copy.deepcopy(fitted.attrs["sem_state"])
    forged["results"]["covariance"] = []
    rehash(forged)
    model = model.model_copy(update={"payload": forged})
    monkeypatch.setattr(
        SEMState, "model_dump", lambda *a, **k: pytest.fail("copy before admission")
    )
    import openecon.econometrics.latent.sem as module

    thaw = module._thaw

    def guarded_thaw(value):
        if isinstance(value, dict) and "results" in value:
            pytest.fail("payload thaw before admission")
        return thaw(value)

    monkeypatch.setattr(module, "_thaw", guarded_thaw)
    with pytest.raises(AnalysisError, match="shape"):
        latent_sem_restore(model)


@pytest.mark.parametrize(
    "field",
    [
        "covariance",
        "standardized_covariance",
        "paths",
        "standardized_paths",
        "disturbance_covariance",
        "node_covariance",
        "implied_covariance",
        "sample_covariance_ml",
        "natural_jacobian",
        "standardized_jacobian",
        "observed_information_solver",
        "parameters",
        "standardized_parameters",
        "information_eigenvalues",
        "score_solver",
        "scales",
        "moment_jacobian_singular_values",
        "intercepts",
        "node_means",
        "sample_means",
    ],
)
def test_all_cached_shapes_are_admitted_before_tensor_replay(fitted, monkeypatch, field):
    import openecon.econometrics.latent.sem as module

    state = copy.deepcopy(fitted.attrs["sem_state"])
    state["results"][field] = []
    rehash(state)
    monkeypatch.setattr(module, "_tensor", lambda *a, **k: pytest.fail("tensor before admission"))
    monkeypatch.setattr(module, "_evaluate", lambda *a, **k: pytest.fail("AD before admission"))
    with pytest.raises((AnalysisError, ValidationError)):
        latent_sem_restore(state)


@pytest.mark.parametrize("value", [True, float("inf"), 2**54 + 1, 2**1024])
def test_bad_cached_primitives_are_admitted_before_tensor_replay(fitted, monkeypatch, value):
    import openecon.econometrics.latent.sem as module

    state = copy.deepcopy(fitted.attrs["sem_state"])
    state["results"]["covariance"][0][0] = value
    # A direct caller mapping is admitted before digest encoding as well.
    monkeypatch.setattr(module, "_tensor", lambda *a, **k: pytest.fail("tensor before admission"))
    with pytest.raises((AnalysisError, ValidationError)):
        latent_sem_restore(state)


def test_unrepresentable_integer_source_moments_refuse_without_overflow(sample, monkeypatch):
    import openecon.econometrics.latent.sem as module

    state = copy.deepcopy(sample[1].attrs["sem_state"])
    state["source"]["rows"][0][0] = 2**1024
    rehash(state)
    monkeypatch.setattr(module, "_tensor", lambda *a, **k: pytest.fail("tensor before admission"))
    with pytest.raises((AnalysisError, ValidationError)):
        latent_sem_restore(state)
    with pytest.raises(AnalysisError, match="finite"):
        latent_sem_covariance(
            [[2**1024, 0], [0, 1]], columns=["x", "y"], n=10, divisor="n", meanstructure=False
        )
    with pytest.raises(AnalysisError, match="finite"):
        latent_sem_covariance(
            [[1, 0], [0, 1]], columns=["x", "y"], n=10, divisor="n", means=[2**1024, 0]
        )


def test_unavailable_cached_ordinary_index_refuses_before_tensor_replay(fitted, monkeypatch):
    import openecon.econometrics.latent.sem as module

    state = copy.deepcopy(fitted.attrs["sem_state"])
    state["results"]["diagnostics"]["TLI"] = None
    rehash(state)
    monkeypatch.setattr(module, "_tensor", lambda *a, **k: pytest.fail("tensor before admission"))
    with pytest.raises((AnalysisError, ValidationError), match="finite"):
        latent_sem_restore(state)


def test_ambient_dtype_device_and_rng_preserved(fitted):
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    rng = torch.random.get_rng_state().clone()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        result = summary()
        restored = latent_sem_restore(result)
        effects = sem_effects(restored, source="z", target="y2", standardized=True)
        scores = latent_scores(restored, pd.DataFrame([[0.0] * 6], columns=COLS))
        assert effects["effects"].shape == (3, 6)
        assert scores.shape == (1, 2)
        assert torch.get_default_dtype() == torch.float32
        assert torch.get_default_device().type == "meta"
        assert torch.equal(torch.random.get_rng_state(), rng)
        np.testing.assert_allclose(result["parameters"], fitted["parameters"], rtol=1e-10)
    finally:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize("case", ["cycle", "means", "factors", "disturbance_confounding"])
def test_unidentified_and_cyclic_models_refuse(case):
    if case == "cycle":
        options = {"paths": {"z": {"a": None}, "a": {"z": None}}}
    elif case == "means":
        options = {"intercepts": {"z": None}}
    elif case == "factors":
        options = {"factors": {"z": COLS, "a": COLS}}
    else:
        options = {"residual_covariances": {("z", "a"): None}}
    arguments = dict(
        columns=COLS,
        n=303,
        divisor="n",
        means=[0.0] * 6,
        factors=FACTORS,
        paths=PATHS,
        tolerance=1e-8,
    )
    arguments.update(options)
    with pytest.raises(AnalysisError):
        latent_sem_covariance(OMEGA[:6, :6].tolist(), **arguments)


@pytest.mark.parametrize(
    "options",
    [
        {"identification": "unit_variance"},
        {"meanstructure": 1},
        {"equalities": {"loading:x2<-z": "both", "variance:x1": "both"}},
        {"fixed": {"loading:x1<-z": 0.5}},
        {"fixed": {"variance:x1": 0.0}},
        {"residual_covariances": {("x1", "x2"): None, ("x2", "x1"): None}},
        {"paths": {"x1": {"z": None}}},
        {"paths": {"a": {"a": None}}},
        {"max_iterations": True},
        {"tolerance": float("nan")},
    ],
)
def test_invalid_specification_domains_refuse(options):
    arguments = dict(
        columns=COLS, n=303, divisor="n", means=[0.0] * 6, factors=FACTORS, paths=PATHS
    )
    arguments.update(options)
    with pytest.raises(AnalysisError):
        latent_sem_covariance(OMEGA[:6, :6].tolist(), **arguments)


def test_summary_means_labels_asymmetry_and_nonpositive_covariance_refuse():
    cases = [
        dict(means=None),
        dict(means=[0.0, 0.0]),
        dict(divisor="auto"),
        dict(covariance=np.ones((6, 6)).tolist()),
        dict(covariance=np.eye(6).tolist(), means=[True] * 6),
        dict(covariance=pd.DataFrame(OMEGA[:6, :6], index=COLS[::-1], columns=COLS)),
        dict(means=pd.Series([0.0] * 6, index=COLS[::-1])),
    ]
    tiny_asymmetric = OMEGA[:6, :6] * 1e-16
    tiny_asymmetric[0, 1] += 1e-18
    cases.append(dict(covariance=tiny_asymmetric.tolist()))
    for case in cases:
        arguments = dict(
            covariance=OMEGA[:6, :6].tolist(),
            columns=COLS,
            n=303,
            divisor="n",
            means=[0.0] * 6,
            factors=FACTORS,
            paths=PATHS,
        )
        arguments.update(case)
        with pytest.raises(AnalysisError):
            latent_sem_covariance(**arguments)


def test_generators_oversized_index_and_unsafe_integers_refuse():
    with pytest.raises(AnalysisError):
        latent_sem(
            {name: (i for i in range(30)) for name in COLS},
            columns=COLS,
            factors=FACTORS,
            paths=PATHS,
        )
    frame = pd.DataFrame(np.random.default_rng(3).normal(size=(20, 6)), columns=COLS)
    frame["x1"] = np.arange(2**53 + 1, 2**53 + 21, dtype=np.int64)
    with pytest.raises(AnalysisError, match="float64"):
        latent_sem(frame, columns=COLS, factors=FACTORS, paths=PATHS)
    frame.index = pd.Index(["a" * 1025 + str(i) for i in range(20)])
    with pytest.raises(AnalysisError, match="index"):
        latent_sem(frame, columns=COLS, factors=FACTORS, paths=PATHS)


def test_mapping_series_alignment_and_score_row_mismatch_refuse(sample):
    frame, result = sample
    mapping = {name: frame[name] for name in COLS}
    mapping[COLS[-1]] = mapping[COLS[-1]].set_axis(frame.index[::-1])
    with pytest.raises(AnalysisError, match="indexes"):
        latent_sem(mapping, columns=COLS, factors=FACTORS, paths=PATHS, missing="drop")
    with pytest.raises(AnalysisError, match="missing"):
        latent_scores(result, frame)
    scores = latent_scores(result, frame, missing="drop")
    assert scores.iloc[3].isna().all()


def test_summary_training_scores_refuse_and_saved_centering_is_required(fitted):
    with pytest.raises(AnalysisError, match="Summary"):
        latent_scores(fitted)
    covariance_only = latent_sem_covariance(
        OMEGA[:6, :6].tolist(),
        columns=COLS,
        n=303,
        divisor="n",
        factors=FACTORS,
        paths=PATHS,
        meanstructure=False,
    )
    with pytest.raises(AnalysisError, match="centering"):
        latent_scores(covariance_only, pd.DataFrame([[0.0] * 6], columns=COLS))


def test_budgeted_nonconvergence_has_no_fabricated_inference():
    with pytest.raises(AnalysisError, match="start"):
        summary(max_iterations=1)


def test_fixed_covariance_has_a_bounded_admissible_start():
    covariance = [[2.0, 1.7, 0.0], [1.7, 1.5, 0.0], [0.0, 0.0, 1.0]]
    result = latent_sem_covariance(
        covariance,
        columns=["a", "b", "c"],
        n=300,
        divisor="n",
        means=[0.0, 0.0, 0.0],
        residual_covariances={("a", "b"): 1.7},
        tolerance=1e-8,
    )
    np.testing.assert_allclose(result["implied_covariance"], covariance, rtol=2e-7, atol=2e-7)
    assert result.attrs["sem_state"]["starts"][0]["variance_rescalings"] > 0


def test_true_zero_residual_variance_refuses_ordinary_wald():
    load = np.array([1.0, 0.8, 0.7])
    covariance = np.outer(load, load) + np.diag([0.0, 0.3, 0.4])
    with pytest.raises(AnalysisError):
        latent_sem_covariance(
            covariance.tolist(),
            columns=["a", "b", "c"],
            factors={"f": ["a", "b", "c"]},
            n=500,
            divisor="n",
            means=[0.0, 0.0, 0.0],
            tolerance=1e-8,
        )


@pytest.mark.parametrize("name", ["a:b", "a,b", "a<-b"])
def test_reserved_name_delimiters_are_refused(name):
    with pytest.raises(AnalysisError):
        latent_sem_covariance(
            np.eye(3).tolist(), columns=[name, "b", "c"], n=20, divisor="n", means=[0.0] * 3
        )


def test_summary_preserves_exact_declared_source_divisor(fitted):
    original = np.array(OMEGA[:6, :6]) * 303 / 302
    result = latent_sem_covariance(
        original.tolist(),
        columns=COLS,
        n=303,
        divisor="n-1",
        means=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        factors=FACTORS,
        paths=PATHS,
        tolerance=1e-8,
    )
    assert result.attrs["sem_state"]["source"]["covariance"] == original.tolist()
    with pytest.raises(AnalysisError, match="float64"):
        latent_sem_covariance(
            [[2**53 + 1, 0, 0], [0, 1, 0], [0, 0, 1]],
            columns=["a", "b", "c"],
            n=20,
            divisor="n",
            means=[0.0] * 3,
        )
