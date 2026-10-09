"""Compact CF reload, source identity and independent query continuity."""
from __future__ import annotations

import json
import os
import shutil
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.control_function.state import digest, validate_state
from openecon.econometrics.control_function.stream_state import bind_source, SCHEMA
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget

KINDS = ("cfregress", "cflogit", "cfprobit", "cfcloglog", "cfpoisson", "cfgamma", "cfinvgauss", "cffraclogit")


def frame(kind="cfregress", n=180, *, scale=1.0, clusters=False):
    rng = np.random.default_rng(823)
    x, z, u, error = rng.normal(size=(4, n))
    d = .5 * x + .8 * z + u
    eta = .1 + .3 * x + .2 * d + .25 * u
    if kind == "cfregress":
        y = scale * (eta + error)
    elif kind in {"cflogit", "cfprobit", "cfcloglog"}:
        y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    elif kind == "cfpoisson":
        y = rng.poisson(np.exp(eta))
    elif kind in {"cfgamma", "cfinvgauss"}:
        y = np.exp(eta) * rng.gamma(3, 1 / 3, n)
    else:
        y = 1 / (1 + np.exp(-(eta + .5 * error)))
    result = pd.DataFrame(dict(y=y, d=d, x=x, z=z))
    if clusters:
        result["cluster"] = np.arange(n) % 30
    return result


def fitted(data, kind="cfregress", *, batch_rows=31, clusters=False):
    return getattr(oe, kind)(data=data, y="y", endogenous="d", x=["x"], instruments=["z"],
                             missing="drop", batch_rows=batch_rows,
                             **({"covariance": "cluster", "cluster": "cluster"} if clusters else {}))


def rehash(result):
    state = result.extra["control_function_state"]
    state["integrity_sha256"] = digest(state)
    result.provenance["control_function_state_sha256"] = state["integrity_sha256"]
    return result


def regenerate_at_beta(result, source, beta, *, batch_rows=13):
    """Coherently replace every equation summary/report without fitting again."""
    from openecon.econometrics.core import wald_test
    from openecon.econometrics.control_function.stream_state import MATRICES, VECTORS
    from openecon.econometrics.streaming_control_function import prepare_stream, evaluate_stream
    from openecon.engines.inference import critical_value, two_sided_p_values
    state = result.extra["control_function_state"]
    gamma = torch.tensor(state["gamma"], dtype=torch.float64)
    evaluated = evaluate_stream(prepare_stream(result.spec, source, batch_rows=batch_rows),
                                gamma, beta, state["kind"])
    for name in (*MATRICES, *VECTORS):
        value = evaluated[name]
        state[name] = None if value is None else value.tolist()
    state["criterion"] = float(evaluated["criterion"])
    covariance = evaluated["joint_covariance"]
    parameters = torch.cat((gamma, beta))
    standard_errors = covariance.diagonal().sqrt()
    statistics = parameters / standard_errors
    p_values = two_sided_p_values(statistics, None)
    critical = critical_value(result.spec.alpha, None)
    for i, coefficient in enumerate(result.coefficients):
        for name, values in (("estimate", parameters), ("std_error", standard_errors),
                              ("statistic", statistics), ("p_value", p_values),
                              ("ci_low", parameters - critical * standard_errors),
                              ("ci_high", parameters + critical * standard_errors)):
            setattr(coefficient, name, float(values[i]))
    result.covariance_matrix = covariance.tolist()
    result.tests = {"control_coefficient_zero": wald_test(parameters, covariance, [len(parameters) - 1])}
    result.metrics["working_criterion"] = state["criterion"]
    result.predictions = evaluated["preview"]
    return rehash(result)


def clone(result, source):
    return bind_source(ResultBundle.model_validate_json(result.model_dump_json()), source)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("clusters", [False, True])
def test_complete_reload_and_common_targets_without_fit(kind, clusters, monkeypatch):
    data = frame(kind, clusters=clusters)
    data.loc[[2, 11], "z"] = np.nan
    data.index = pd.Index((["same", 1, None] * 60), dtype=object, name="row")
    source = Dataset.from_frame(data)
    result = fitted(source, kind, clusters=clusters)
    assert result.extra["control_function_state"]["schema"] == SCHEMA
    before = result.model_dump_json()
    query = data.iloc[:19]
    expected = oe.predict(result, query, interval="mean")
    effects = oe.margins(result, ["x", "d", "z"], data=query, method="mem")
    from openecon.econometrics import streaming_control_function as engine
    from openecon.econometrics.control_function import kernels
    def forbidden(*args, **kwargs):
        pytest.fail("Saved-state evaluation optimized or fit a model.")
    monkeypatch.setattr(engine, "solve_stream", forbidden)
    monkeypatch.setattr(engine, "maximize_newton", forbidden)
    monkeypatch.setattr(kernels, "fit_joint", forbidden)
    restored = oe.cf_restore(result=before, data=source, batch_rows=17)
    actual = oe.predict(restored, query, interval="mean")
    pd.testing.assert_frame_equal(actual, expected, rtol=2e-9, atol=2e-11)
    actual_effects = oe.margins(restored, ["x", "d", "z"], data=query, method="mem")
    np.testing.assert_allclose(actual_effects.select_dtypes("number"), effects.select_dtypes("number"), rtol=2e-9, atol=2e-11)
    assert restored.model_dump_json() == before
    assert result.model_dump_json() == before
    assert "_control_function_source" not in restored.model_dump()


def test_factory_reload_needs_explicit_source_and_compact_rows_are_absent():
    data = frame(n=900)
    source = Dataset.from_batches(lambda: (data.iloc[i:i+29] for i in range(0, len(data), 29)),
                                  list(data.columns), row_count=len(data))
    result = fitted(source)
    state = result.extra["control_function_state"]
    assert state["source"] is None and result.sample_positions == []
    assert not ({"z", "x", "d", "y", "residual", "design", "fitted", "row_scores",
                 "source_index", "source_columns", "sample_positions", "cluster_codes"} & set(state))
    assert len(json.dumps(state)) < 30_000
    saved = ResultBundle.model_validate_json(result.model_dump_json())
    with pytest.raises(AnalysisError, match="cf_restore"):
        oe.predict(saved, data.iloc[:2])
    with pytest.raises(AnalysisError, match="cf_restore"):
        oe.cf_restore(result=result.model_dump_json())
    restored = oe.cf_restore(result=result.model_dump_json(), data=source)
    assert len(oe.predict(restored, data.iloc[:2])) == 2


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_physical_json_auto_reload_changed_source_and_explicit_relocation(tmp_path, suffix):
    data = frame(n=230)
    path = tmp_path / ("training." + suffix)
    if suffix == "csv":
        data.to_csv(path, index=False)
    else:
        data.index = pd.date_range("2026-01-01", periods=len(data), tz="UTC", name="when")
        data.to_parquet(path)
    source = scan(path)
    result = fitted(source)
    saved = ResultBundle.model_validate_json(result.model_dump_json())
    expected = oe.predict(result, data.iloc[:8], interval="mean")
    actual = oe.predict(saved, data.iloc[:8], interval="mean")
    pd.testing.assert_frame_equal(actual, expected, rtol=2e-9, atol=2e-11)
    relocated = tmp_path / ("relocated." + suffix)
    shutil.copyfile(path, relocated)
    path.unlink()
    moved = oe.cf_restore(result=result.model_dump_json(), data=scan(relocated))
    np.testing.assert_allclose(oe.predict(moved, data.iloc[:8]), expected.iloc[:, :1], rtol=2e-9, atol=2e-11)
    damaged = data.copy()
    damaged.loc[damaged.index[0], "z"] += .1
    if suffix == "csv":
        damaged.to_csv(path, index=False)
    else:
        damaged.to_parquet(path)
    with pytest.raises(AnalysisError, match="changed"):
        oe.predict(ResultBundle.model_validate_json(result.model_dump_json()), data.iloc[:2])


@pytest.mark.parametrize("changed", ["value", "index", "cluster"])
def test_explicit_source_binding_refuses_same_count_mutation(changed):
    data = frame(clusters=True)
    data["cluster"] = pd.Series(([True, 1, False, 0, *[f"c{i}" for i in range(26)]] * 6), dtype=object)
    source = Dataset.from_frame(data)
    result = fitted(source, clusters=True)
    assert result.inference["cluster_count"] == 30
    altered = data.copy(deep=True)
    if changed == "value":
        altered.loc[0, "z"] += .1
    elif changed == "index":
        altered.index = pd.RangeIndex(1, len(altered) + 1)
    else:
        altered.loc[0, "cluster"] = 1  # Equal under pandas factorize, distinct under CF typing.
    with pytest.raises(AnalysisError, match="changed"):
        oe.cf_restore(result=result.model_dump_json(), data=Dataset.from_frame(altered))


@pytest.mark.parametrize("name", ["gamma", "beta", "bread", "meat", "joint_covariance", "score_sum",
                                  "score_abs_sum", "first_stage_factor", "first_stage_target", "outcome_factor", "outcome_target"])
def test_coherently_rehashed_compact_summaries_refuse(name):
    data = frame()
    source = Dataset.from_frame(data)
    result = clone(fitted(source), source)
    value = result.extra["control_function_state"][name]
    if isinstance(value[0], list):
        value[0][0] += .2
    else:
        value[0] += .2
    rehash(result)
    with pytest.raises(AnalysisError):
        oe.predict(result, data.iloc[:2])


def test_compact_helper_retains_joint_jacobians_and_link_intervals():
    data = frame("cflogit")
    result = fitted(Dataset.from_frame(data), "cflogit")
    output = oe.cf_predict(result=result, data=data.iloc[:9], alpha=.1)
    critical = 1.6448536269514722
    expected = 1 / (1 + np.exp(-(output.eta - critical * output.eta_std_error)))
    np.testing.assert_allclose(output.ci_low, expected, rtol=1e-12, atol=1e-14)
    assert len(output.attrs["mean_jacobian"]) == 9
    assert output.attrs["full_gamma_beta_covariance"] is True
    assert output.attrs["source_semantic_replay"] is True
    with pytest.raises(AnalysisError, match="resident query"):
        oe.cf_predict(result=result)


@pytest.mark.parametrize("changed", ["preview", "optimizer", "inference", "positions", "binding", "path", "wald"])
def test_rehashed_reporting_source_and_semantics_refuse(changed):
    data = frame()
    source = Dataset.from_frame(data)
    result = clone(fitted(source), source)
    state = result.extra["control_function_state"]
    if changed == "preview":
        result.predictions[0]["fitted"] += .2
    elif changed == "optimizer":
        state["optimizer"]["tolerance"] = .1
        result.provenance["optimizer"] = dict(state["optimizer"])
    elif changed == "inference":
        result.inference["small_sample_correction"] = 2.
    elif changed == "positions":
        state["sample_positions_hash"] = "0" * 64
        result.provenance["sample_positions_hash"] = "0" * 64
    elif changed == "binding":
        state["source_binding"]["typed_index_hash"] = "0" * 64
    elif changed == "path":
        state["source"] = {"kind": "csv", "path": "https://invalid.example/training.csv"}
    else:
        result.tests["control_coefficient_zero"]["p_value"] = .01
    rehash(result)
    with pytest.raises(AnalysisError):
        oe.predict(result, data.iloc[:2])


def test_factory_index_mutation_between_replay_passes_refuses():
    data = frame()
    count = [0]
    def factory():
        count[0] += 1
        if count[0] > 1:
            changed = data.copy(deep=False)
            changed.index = pd.RangeIndex(1, len(data) + 1)
            yield changed
        else:
            yield data
    source = Dataset.from_batches(factory, list(data.columns), row_count=len(data))
    with pytest.raises(AnalysisError, match="changed"):
        fitted(source, batch_rows=len(data))


def test_physical_cache_reads_all_bytes_and_all_reporting_before_hit(tmp_path, monkeypatch):
    data = frame(n=600)
    path = tmp_path / "training.csv"
    data.to_csv(path, index=False)
    source = scan(path)
    result = fitted(source)
    from openecon.econometrics import streaming_control_function as engine
    original, calls = engine.evaluate_stream, []
    def evaluate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "evaluate_stream", evaluate)
    validate_state(result)
    assert len(calls) == 1
    assert "_control_function_semantic_cache" not in result.model_dump()
    validate_state(result)
    oe.predict(result, data.iloc[:4])
    oe.margins(result, "d", data=data.iloc[:7])
    assert len(calls) == 1
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="budget"):
        validate_state(result)
    assert len(calls) == 1
    # Changing top-level reporting invalidates the token although state SHA is
    # unchanged. The canonical scientific report is then checked afresh.
    result.tests["control_coefficient_zero"]["p_value"] = .01
    with pytest.raises(AnalysisError, match="Wald"):
        validate_state(result)
    assert len(calls) == 2


def test_physical_cache_rehashed_state_and_forged_stat_identity_refuse(tmp_path, monkeypatch):
    data = frame(n=500)
    path = tmp_path / "training.csv"
    data.to_csv(path, index=False)
    source = scan(path)
    result = fitted(source)
    validate_state(result)
    original = result.model_dump_json()
    result.extra["control_function_state"]["bread"][0][0] += .2
    rehash(result)
    with pytest.raises(AnalysisError, match="bread"):
        validate_state(result)
    restored = oe.cf_restore(result=original, data=source)
    before = path.stat()
    payload = bytearray(path.read_bytes())
    # Alter an outcome value without changing file length or its mtime.
    offset = payload.index(b"\n") + 1
    while payload[offset] not in b"0123456789":
        offset += 1
    payload[offset] = ord("9") if payload[offset] != ord("9") else ord("8")
    path.write_bytes(payload)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    # Bypass cheap stat identity to prove the private cache still binds all
    # bytes. A stale mtime/size is never sufficient for the fast return.
    monkeypatch.setattr(source, "assert_unchanged", lambda: None)
    with pytest.raises(AnalysisError):
        validate_state(restored)


def test_reported_covariance_shape_refused_before_source_allocation(monkeypatch):
    data = frame()
    source = Dataset.from_frame(data)
    result = fitted(source)
    result.covariance_matrix[0].append(0.)
    def forbidden(*args, **kwargs):
        pytest.fail("Malformed covariance reached source or tensor replay.")
    monkeypatch.setattr(source, "iter_batches", forbidden)
    with pytest.raises(AnalysisError, match="fixed-width"):
        validate_state(result)


@pytest.mark.parametrize("changed", ["iterations", "refinement", "step", "definition", "scaled_gradient", "concavity"])
def test_rehashed_optimizer_history_and_refinement_refuse(changed):
    data = frame("cflogit")
    source = Dataset.from_frame(data)
    result = clone(fitted(source, "cflogit"), source)
    optimizer = result.extra["control_function_state"]["optimizer"]
    if changed == "iterations":
        optimizer["iterations"] = optimizer["max_iterations"] + 1
    elif changed == "refinement":
        optimizer["refined_scaled_gradient_tolerance"] *= 2
    elif changed == "step":
        optimizer["refined_step_tolerance"] *= 2
    elif changed == "definition":
        optimizer["stationarity_refinement"] = "unspecified"
    elif changed == "scaled_gradient":
        optimizer["scaled_gradient"] = 2 * optimizer["refined_scaled_gradient_tolerance"]
    else:
        optimizer["concave"] = False
    result.provenance["optimizer"] = dict(optimizer)
    rehash(result)
    with pytest.raises(AnalysisError, match="optimizer"):
        validate_state(result)


def test_single_list_cluster_saved_semantics_replay():
    data = frame(clusters=True)
    source = Dataset.from_frame(data)
    result = oe.cfregress(data=source, y="y", endogenous="d", x=["x"], instruments=["z"],
                          covariance="cluster", cluster=["cluster"], batch_rows=31)
    restored = oe.cf_restore(result=result.model_dump_json(), data=source)
    assert restored.inference["cluster_column"] == "cluster"
    assert restored.inference["cluster_count"] == 30
    assert len(oe.predict(restored, data.iloc[:3])) == 3


@pytest.mark.parametrize("changed", ["missing", "extra", "requested", "device", "precision", "metal",
                                     "fallbacks", "empty", "offload", "bool_count", "zero_count",
                                     "negative_count", "float_count"])
def test_execution_provenance_refused_before_source_replay(changed, monkeypatch):
    source = Dataset.from_frame(frame())
    result = fitted(source)
    execution = result.provenance["execution"]
    if changed == "missing":
        del execution["metal_precision"]
    elif changed == "extra":
        execution["unverified_acceleration"] = True
    elif changed == "requested":
        execution["requested_device"] = "cuda"
    elif changed == "device":
        execution["device"] = "cuda"
    elif changed == "precision":
        execution["reporting_precision"] = "float32"
    elif changed == "metal":
        execution["metal_precision"] = "float32 preconditioner"
    elif changed == "fallbacks":
        execution["fallbacks"] = {"cuda_unavailable": 1}
    elif changed == "empty":
        execution["factor_devices"] = {}
    elif changed == "offload":
        execution["factor_devices"] = {"cuda": 999}
    else:
        execution["factor_devices"] = {"cpu": {"bool_count": True, "zero_count": 0,
                                              "negative_count": -1, "float_count": 1.5}[changed]}
    from openecon.econometrics import streaming_control_function as engine
    def forbidden(*args, **kwargs):
        pytest.fail("Contradictory execution metadata reached source replay.")
    monkeypatch.setattr(engine, "prepare_stream", forbidden)
    with pytest.raises(AnalysisError, match="execution provenance"):
        oe.cf_restore(result=result.model_dump_json(), data=source)


def test_physical_cache_cannot_admit_changed_execution_provenance(tmp_path, monkeypatch):
    path = tmp_path / "training.csv"
    frame().to_csv(path, index=False)
    source = scan(path)
    result = fitted(source)
    validate_state(result)
    validate_state(result)
    result.provenance["execution"]["factor_devices"] = {"cpu": True}
    from openecon.econometrics import streaming_control_function as engine
    monkeypatch.setattr(engine, "evaluate_stream", lambda *args, **kwargs: pytest.fail("Invalid cache metadata replayed."))
    with pytest.raises(AnalysisError, match="execution provenance"):
        validate_state(result)


def test_coherently_regenerated_nonstationary_outcome_refuses_without_refit(monkeypatch):
    source = Dataset.from_frame(frame("cflogit"))
    result = fitted(source, "cflogit")
    beta = torch.tensor(result.extra["control_function_state"]["beta"], dtype=torch.float64)
    beta[0] += 1e-7
    regenerate_at_beta(result, source, beta)
    from openecon.econometrics import streaming_control_function as engine
    monkeypatch.setattr(engine, "maximize_newton", lambda *args, **kwargs: pytest.fail("Validation refit the outcome."))
    with pytest.raises(AnalysisError, match="strict stationarity refinement"):
        oe.cf_restore(result=result.model_dump_json(), data=source, batch_rows=17)


@pytest.mark.parametrize("kind", KINDS[1:])
def test_refinement_allows_one_ulp_parameters_and_rebatched_replay(kind):
    source = Dataset.from_frame(frame(kind))
    result = fitted(source, kind)
    beta = torch.tensor(result.extra["control_function_state"]["beta"], dtype=torch.float64)
    beta[0] = torch.nextafter(beta[0], torch.tensor(float("inf"), dtype=torch.float64))
    regenerate_at_beta(result, source, beta)
    restored = oe.cf_restore(result=result.model_dump_json(), data=source, batch_rows=17)
    assert restored.extra["control_function_state"]["beta"] == beta.tolist()
    optimizer = restored.extra["control_function_state"]["optimizer"]
    assert optimizer["scaled_gradient"] <= optimizer["refined_scaled_gradient_tolerance"]


def test_admission_precedes_source_and_ambient_state_is_preserved(monkeypatch):
    data = frame()
    source = Dataset.from_frame(data)
    result = fitted(source)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="budget"):
        validate_state(result)
    with pytest.raises(AnalysisError, match="max_work"):
        oe.cf_restore(result=result, data=source, max_work=1)
    rng, dtype = torch.get_rng_state().clone(), torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            validate_state(result)
            output = oe.cf_predict(result=result, data=data.iloc[:2])
            assert torch.empty(0).device.type == "meta"
        assert torch.get_default_dtype() == torch.float32
        assert torch.equal(torch.get_rng_state(), rng)
        assert len(output) == 2
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("representation", ["object", "mapping", "json"])
@pytest.mark.parametrize("auxiliary", ["strings", "breadth"])
def test_restore_complete_reporting_admitted_before_any_model_copy(compact, representation, auxiliary, monkeypatch):
    data = frame(n=24)
    source = Dataset.from_frame(data)
    result = (fitted(source) if compact else
              oe.cfregress(data=data, y="y", endogenous="d", x=["x"], instruments=["z"]))
    result.extra["unbounded_reporting"] = (["a" * 65_000 for _ in range(70)] if auxiliary == "strings"
                                            else [None] * 200_000)
    supplied = (result if representation == "object" else result.model_dump(mode="json")
                if representation == "mapping" else result.model_dump_json())
    def forbidden(*args, **kwargs):
        pytest.fail("Complete reporting was copied before workspace admission.")
    monkeypatch.setattr(ResultBundle, "model_dump", forbidden)
    monkeypatch.setattr(ResultBundle, "model_dump_json", forbidden)
    monkeypatch.setattr(ResultBundle, "model_copy", forbidden)
    monkeypatch.setattr(ResultBundle, "model_validate", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="budget|workspace"):
        oe.cf_restore(result=supplied, **({"data": source} if compact else {}))


def test_wide_metadata_walker_does_not_allocate_a_wide_traversal_stack():
    from openecon.econometrics.control_function.stream_state import _metadata_size
    caller_owned = {"auxiliary": [None] * 200_000}
    tracemalloc.start()
    try:
        with use_workspace_budget(1), pytest.raises(AnalysisError, match="budget|workspace"):
            _metadata_size(caller_owned)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 512 * 1024


@pytest.mark.parametrize("representation", ["object", "mapping", "json"])
def test_legacy_restore_retains_reporting_and_prediction(representation):
    data = frame(n=80)
    result = oe.cfregress(data=data, y="y", endogenous="d", x=["x"], instruments=["z"])
    before = result.model_dump_json()
    supplied = (result if representation == "object" else result.model_dump(mode="json")
                if representation == "mapping" else before)
    restored = oe.cf_restore(result=supplied)
    assert restored.model_dump_json() == before
    pd.testing.assert_frame_equal(oe.cf_predict(result=restored), oe.cf_predict(result=result))
