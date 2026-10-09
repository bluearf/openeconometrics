"""Full two-stage scientific and resource contracts for the replay engine."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.core import make_spec
from openecon.econometrics.replay_sample import ReplaySample
from openecon.econometrics.streaming_control_function import prepare_stream, solve_stream, evaluate_stream
from openecon.resources import use_workspace_budget

ROOT = Path(__file__).resolve().parents[1]
MODULE = importlib.util.spec_from_file_location("stream_cf_independent_oracle", ROOT/"scripts/verify_control_function_oracles.py")
oracle = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(oracle)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def specification(kind, inputs, *, cluster=None, **options):
    return make_spec(oracle.APIS[kind], outcome=inputs["y"], predictors=inputs["x"],
                     columns={"endogenous": inputs["endogenous"], "instruments": inputs["instruments"]},
                     covariance="cluster" if cluster else "robust", cluster=cluster,
                     intercept=inputs["intercept"], missing=inputs["missing"],
                     options={"max_iterations": 25, **options})


@pytest.mark.parametrize("kind", oracle.KINDS)
@pytest.mark.parametrize("cluster", [None, "cluster"])
def test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences(kind, cluster):
    inputs = oracle.fixture(kind)
    inputs["cluster"] = cluster
    frame = pd.DataFrame(inputs["data"])
    before = frame.copy(deep=True)
    sample, solved = solve_stream(specification(kind, inputs, cluster=cluster), Dataset.from_frame(frame), batch_rows=17)
    reference = oracle.fit_oracle(kind, inputs)
    np.testing.assert_allclose(solved["gamma"], reference["gamma"], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(solved["beta"], reference["beta"], rtol=2e-7, atol=2e-7)
    evaluated = oracle.reference_at(kind, oracle.prepare_inputs(inputs), solved["gamma"].numpy(), solved["beta"].numpy())
    np.testing.assert_allclose(solved["bread"], evaluated["numeric_bread"], rtol=2e-7, atol=2e-7)
    for name in ("bread", "meat", "joint_covariance"):
        np.testing.assert_allclose(solved[name], evaluated[name], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(solved["criterion"], evaluated["criterion"], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(solved["score_sum"], evaluated["row_scores"].sum(0), rtol=1e-6, atol=2e-11)
    first = len(solved["gamma"])
    assert np.max(np.abs(solved["bread"].numpy()-solved["bread"].numpy().T)) > 1
    assert np.max(np.abs(solved["joint_covariance"].numpy()[:first, first:])) > 1e-5
    assert torch.equal(solved["bread"][:first, first:], torch.zeros((first, len(solved["beta"])), dtype=torch.float64))
    assert solved["optimizer"]["converged"]
    assert sample.actual_numeric_peak_rows <= 17
    assert sample.cf_original_source.row_count == len(frame)
    assert solved["cluster_count"] == (len(frame)//4 if cluster else None)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("kind", oracle.KINDS)
def test_explicit_saved_parameter_replay_does_not_refit_and_is_batch_independent(kind, monkeypatch):
    inputs = oracle.fixture(kind)
    spec = specification(kind, inputs)
    frame = pd.DataFrame(inputs["data"])
    sample, solved = solve_stream(spec, Dataset.from_frame(frame), batch_rows=13)
    restored = prepare_stream(spec, Dataset.from_frame(frame), batch_rows=31)
    import openecon.econometrics.streaming_control_function as engine
    monkeypatch.setattr(engine, "least_squares", lambda *a, **k: pytest.fail("Saved parameter replay must not refit either stage"))
    monkeypatch.setattr(engine, "maximize_newton", lambda *a, **k: pytest.fail("Saved parameter replay must not refit either stage"))
    checked = evaluate_stream(restored, solved["gamma"], solved["beta"], kind)
    assert restored.cf_source_binding == sample.cf_source_binding
    assert restored.provenance()["data_hash"] == sample.provenance()["data_hash"]
    assert restored.provenance()["sample_positions_hash"] == sample.provenance()["sample_positions_hash"]
    for name in ("bread", "meat", "joint_covariance", "criterion"):
        np.testing.assert_allclose(checked[name], solved[name], rtol=2e-11, atol=2e-11)
    assert [v["row"] for v in checked["preview"]] == [v["row"] for v in solved["preview"]]
    np.testing.assert_allclose(pd.DataFrame(checked["preview"]), pd.DataFrame(solved["preview"]), rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("kind", oracle.KINDS)
def test_singleton_clusters_reduce_to_hc0_without_cr1(kind):
    inputs = oracle.fixture(kind)
    source = Dataset.from_frame(pd.DataFrame(inputs["data"]))
    _, hc0 = solve_stream(specification(kind, inputs), source, batch_rows=29)
    _, cr0 = solve_stream(specification(kind, inputs, cluster="singleton"), source, batch_rows=29)
    np.testing.assert_allclose(cr0["meat"], hc0["meat"], rtol=3e-13, atol=3e-11)
    np.testing.assert_allclose(cr0["joint_covariance"], hc0["joint_covariance"], rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("kind", oracle.KINDS)
def test_no_intercept_original_coordinate_factors_and_covariance(kind):
    inputs = oracle.fixture(kind)
    inputs["intercept"] = False
    sample, solved = solve_stream(specification(kind, inputs), Dataset.from_frame(pd.DataFrame(inputs["data"])), batch_rows=19)
    assert solved["z_terms"] == ["x", "z1", "z2"]
    assert solved["x_terms"] == ["x", "d"]
    ref = oracle.reference_at(kind, oracle.prepare_inputs(inputs), solved["gamma"].numpy(), solved["beta"].numpy())
    np.testing.assert_allclose(solved["joint_covariance"], ref["joint_covariance"], rtol=3e-11, atol=3e-11)
    z, d = ref["z"], ref["d"]
    factor, target = solved["first_stage_factor"].numpy(), solved["first_stage_target"].numpy()
    np.testing.assert_allclose(factor.T@factor, z.T@z, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(factor.T@target, z.T@d, rtol=2e-12, atol=2e-12)
    assert sample.nrows == len(d)


def test_common_missing_sample_preview_and_typed_cluster_identity():
    inputs = oracle.fixture("gaussian")
    inputs["missing"] = "drop"
    frame = pd.DataFrame(inputs["data"])
    # Distinct groups under CF's typed identity, including Python bool/int.
    labels = [True, 1, False, 0, "a", "b", "c", "d", "e", "f", "g", "h"]
    frame["typed"] = pd.Series([labels[i % len(labels)] for i in range(len(frame))], dtype=object)
    frame.loc[1, "z2"] = np.nan
    frame.loc[4, "y"] = np.nan
    frame.loc[7, "typed"] = None
    sample, solved = solve_stream(specification("gaussian", inputs, cluster="typed"), Dataset.from_frame(frame), batch_rows=13)
    assert sample.nrows == len(frame)-3
    assert solved["cluster_count"] == 12
    assert [p["row"] for p in solved["preview"]] == [i for i in range(len(frame)) if i not in {1, 4, 7}]
    selected = frame.dropna(subset=["x", "z1", "z2", "d", "y", "typed"])
    z = np.column_stack((np.ones(len(selected)), selected[["x", "z1", "z2"]]))
    residual = selected.d.to_numpy()-z@solved["gamma"].numpy()
    q = np.column_stack((np.ones(len(selected)), selected[["x", "d"]], residual))
    s = selected.y.to_numpy()-q@solved["beta"].numpy()
    rows = np.column_stack((z*residual[:, None], q*s[:, None]))
    grouped = {}
    for label, score in zip(selected.typed, rows):
        key = (type(label), label)
        grouped[key] = grouped.get(key, np.zeros(rows.shape[1]))+score
    units = np.stack(list(grouped.values()))
    np.testing.assert_allclose(solved["meat"], units.T@units, rtol=2e-12, atol=2e-10)


@pytest.mark.parametrize("mutation", ["index", "typed_cluster"])
def test_mutating_typed_source_binding_is_rejected_even_if_pandas_value_hash_collides(mutation):
    inputs = oracle.fixture("gaussian")
    frame = pd.DataFrame(inputs["data"])
    frame["typed"] = pd.Series([i % 12 for i in range(len(frame))], dtype=object)
    calls = 0

    def batches():
        nonlocal calls
        calls += 1
        current = frame.copy(deep=True)
        if calls > 1:
            if mutation == "index":
                current.index = pd.RangeIndex(1, len(current)+1)
            else:
                current.loc[0, "typed"] = False  # 0 and False hash alike in pandas.
        for start in range(0, len(current), 17):
            yield current.iloc[start:start+17]
    source = Dataset.from_batches(batches, frame.columns, row_count=len(frame))
    with pytest.raises(AnalysisError, match="Typed source indices or cluster labels changed"):
        solve_stream(specification("gaussian", inputs, cluster="typed"), source, batch_rows=17)


@pytest.mark.parametrize("kind", ["logit", "probit", "cloglog", "fractional_logit"])
def test_complete_and_quasi_separation_and_fractional_interior_constraints(kind):
    inputs = oracle.fixture(kind)
    frame = pd.DataFrame(inputs["data"])
    frame["y"] = (frame.x > 0).astype(float)
    with pytest.raises(AnalysisError, match="separation"):
        solve_stream(specification(kind, inputs), Dataset.from_frame(frame), batch_rows=23)


def test_over_5000_rows_remains_block_bounded_without_source_collection():
    inputs = oracle.fixture("gaussian", rows=6001)
    frame = pd.DataFrame(inputs["data"])
    source = Dataset.from_frame(frame)
    source.collect = lambda *a, **k: pytest.fail("Dataset cannot be collected by CF")
    sample, solved = solve_stream(specification("gaussian", inputs), source, batch_rows=113)
    assert solved["nobs"] == 6001
    assert sample.passes == 7
    assert sample.actual_numeric_peak_rows <= 113
    assert len(solved["preview"]) == 400
    assert len(sample.sample) == 400
    assert solved["first_stage_factor"].shape == (5, 4)
    assert solved["outcome_factor"].shape == (5, 4)
    assert solved["resource_plan"]["estimated_workspace_bytes"] <= solved["resource_plan"]["budget_bytes"]
    assert not any(name in solved for name in ("row_scores", "design", "z", "x", "d", "y", "scalar_score", "scalar_derivative"))


def test_rank_drops_cluster_singularity_work_and_workspace_are_refused():
    inputs = oracle.fixture("gaussian")
    frame = pd.DataFrame(inputs["data"])
    spec = specification("gaussian", inputs)
    singular = frame.copy()
    singular["z2"] = singular.z1
    with pytest.raises(AnalysisError, match="silent rank drops"):
        solve_stream(spec, Dataset.from_frame(singular), batch_rows=23)
    with pytest.raises(AnalysisError, match="work estimate"):
        solve_stream(specification("gaussian", inputs, max_work=1), Dataset.from_frame(frame), batch_rows=23)
    frame["cluster"] = np.arange(len(frame)) % 8
    with pytest.raises(AnalysisError, match="more clusters than joint"):
        solve_stream(specification("gaussian", inputs, cluster="cluster"), Dataset.from_frame(frame), batch_rows=23)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="budget"):
        solve_stream(spec, Dataset.from_frame(frame), batch_rows=8192)


def _stored_source(frame, storage, tmp_path):
    if storage == "frame":
        return Dataset.from_frame(frame)
    path = tmp_path / ("reader." + storage)
    if storage == "csv":
        frame.to_csv(path, index=False, float_format="%.17g")
    else:
        frame.to_parquet(path, row_group_size=257)
    return scan(path)


@pytest.mark.parametrize("storage", ["frame", "csv", "parquet"])
def test_original_producer_obeys_discovery_plan_before_projection_allocation(storage, tmp_path, monkeypatch):
    # The declared 36-coefficient envelope is legal. Its discovery fits 1 MiB,
    # whereas its subsequent global-design reservation correctly refuses it.
    names = ["y", "d", *[f"x{i}" for i in range(8)], *[f"z{i}" for i in range(16)]]
    values = np.random.default_rng(194).normal(size=(8200, len(names)))
    frame = pd.DataFrame(values, columns=names)
    frame["unused"] = 1.0  # The actual producer must allocate the projection.
    source = _stored_source(frame, storage, tmp_path)
    spec = make_spec("cfregress", outcome="y", predictors=names[2:10],
                     columns={"endogenous": "d", "instruments": names[10:]}, covariance="robust")
    requests, produced = [], []
    native = source._raw_batches
    with use_workspace_budget(1):
        admitted = ReplaySample(spec, source)
        assert admitted.reader_rows == 78
        assert admitted.discovery_plan.estimated_bytes <= 1024**2

        def producer(selected, batch_rows):
            # This assertion precedes the frame projection or physical parse;
            # an outer wrapper slicing an 8192-row result cannot satisfy it.
            requests.append(batch_rows)
            assert batch_rows <= admitted.reader_rows
            assert set(selected) == set(names)
            for block in native(selected, batch_rows):
                produced.append(len(block))
                assert len(block) <= admitted.reader_rows
                yield block

        monkeypatch.setattr(source, "_raw_batches", producer)
        with pytest.raises(AnalysisError) as rejected:
            prepare_stream(spec, source)
        assert rejected.value.code == "workspace_limit"
    assert requests and max(requests) == 78
    assert sum(produced) == 8200


@pytest.mark.parametrize("storage", ["frame", "csv", "parquet"])
def test_budgeted_original_reader_and_typed_receipts_survive_complete_replay(storage, tmp_path, monkeypatch):
    inputs = oracle.fixture("gaussian")
    source = _stored_source(pd.DataFrame(inputs["data"]), storage, tmp_path)
    spec = specification("gaussian", inputs)
    requests = []
    native = source._raw_batches
    with use_workspace_budget(2):
        admitted = ReplaySample(spec, source)

        def producer(selected, batch_rows):
            requests.append(batch_rows)
            assert batch_rows <= admitted.reader_rows
            yield from native(selected, batch_rows)

        monkeypatch.setattr(source, "_raw_batches", producer)
        sample, solved = solve_stream(spec, source)
        replay = prepare_stream(spec, source, batch_rows=17)
        checked = evaluate_stream(replay, solved["gamma"], solved["beta"], "gaussian")
    assert requests and max(requests) == admitted.reader_rows < 8192
    assert sample.passes == 7
    assert sample.cf_source_binding == replay.cf_source_binding
    for key in ("data_hash", "sample_hash", "sample_positions_hash"):
        assert sample.provenance()[key] == replay.provenance()[key]
    for key in ("bread", "meat", "joint_covariance", "criterion"):
        np.testing.assert_allclose(solved[key], checked[key], rtol=3e-11, atol=3e-11)


def test_cpu_float64_survives_ambient_default_without_changing_rng():
    inputs = oracle.fixture("gaussian")
    old_dtype, rng = torch.get_default_dtype(), torch.get_rng_state().clone()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            _, solved = solve_stream(specification("gaussian", inputs), Dataset.from_frame(pd.DataFrame(inputs["data"])), batch_rows=23)
            assert torch.empty(0).device.type == "meta"
        assert solved["gamma"].device.type == "cpu" and solved["gamma"].dtype == torch.float64
        assert torch.get_default_dtype() == torch.float32
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        torch.set_default_dtype(old_dtype)


def test_nonconvergence_and_exhausted_separation_budget_refuse_ordinary_inference(monkeypatch):
    inputs = oracle.fixture("logit")
    source = Dataset.from_frame(pd.DataFrame(inputs["data"]))
    with pytest.raises(AnalysisError, match="optimizer did not converge"):
        solve_stream(specification("logit", inputs, max_iterations=1), source, batch_rows=23)
    import openecon.econometrics.streaming_control_function as engine
    monkeypatch.setattr(engine, "CERTIFICATE_MAX_PASSES", 1)
    with pytest.raises(AnalysisError, match="did not converge in 1 passes"):
        solve_stream(specification("logit", inputs), source, batch_rows=23)


def test_rank_condition_uses_original_column_scaled_design_even_after_centering():
    inputs = oracle.fixture("gaussian")
    frame = pd.DataFrame(inputs["data"])
    frame["x"] = 1+1e-10*frame.x
    with pytest.raises(AnalysisError, match="ill-conditioned after column scaling"):
        solve_stream(specification("gaussian", inputs), Dataset.from_frame(frame), batch_rows=23)


@pytest.mark.parametrize("dtype", ["Int64", "int64[pyarrow]", "boolean"])
def test_nullable_cluster_hash_representation_is_independent_of_missing_block(dtype):
    inputs = oracle.fixture("gaussian")
    inputs["missing"] = "drop"
    frame = pd.DataFrame(inputs["data"])
    if dtype == "boolean":
        # A bool cluster alone cannot identify the full joint covariance; use
        # preparation to verify raw binding, not an unsupported two-group fit.
        frame["cluster"] = pd.Series(np.arange(len(frame)) % 2 == 0, dtype=dtype)
    else:
        frame["cluster"] = pd.Series(frame.cluster, dtype=dtype)
    frame.loc[0, "cluster"] = pd.NA
    spec = specification("gaussian", inputs, cluster="cluster")
    first = prepare_stream(spec, Dataset.from_frame(frame), batch_rows=13)
    second = prepare_stream(spec, Dataset.from_frame(frame), batch_rows=31)
    assert first.nrows == second.nrows == len(frame)-1
    assert first.cf_source_binding == second.cf_source_binding
    if dtype != "boolean":
        _, solved = solve_stream(spec, Dataset.from_frame(frame), batch_rows=13)
        assert solved["cluster_count"] == len(frame)//4


@pytest.mark.parametrize("operation", ["prepare", "solve", "evaluate"])
def test_each_cf_engine_entry_forces_cpu_factors_under_ambient_auto_cuda(operation, monkeypatch):
    from openecon.engines import execution

    inputs = oracle.fixture("gaussian")
    spec = specification("gaussian", inputs)
    source = Dataset.from_frame(pd.DataFrame(inputs["data"]))
    # Set up saved parameters before entering the adversarial ambient scope.
    prepared, saved = solve_stream(spec, source, batch_rows=23)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    native_preference = execution._preferred_device
    seen = []

    def cpu_only(trace, block):
        assert trace.requested == "cpu", "CF inherited an ambient automatic/accelerator QR preference"
        chosen = native_preference(trace, block)
        assert chosen == "cpu", "CF attempted to offload a factor"
        seen.append((block.device.type, block.dtype))
        return chosen

    monkeypatch.setattr(execution, "_preferred_device", cpu_only)
    monkeypatch.setattr(execution, "_metal_factor", lambda *args: pytest.fail("CF must never attempt a Metal factor"))
    with execution.execution_scope("auto") as outer:
        assert execution._TRACE.get() is outer and outer.requested == "auto"
        if operation == "prepare":
            current = prepare_stream(spec, source, batch_rows=17)
            metadata = current.cf_execution
        elif operation == "solve":
            _, current = solve_stream(spec, source, batch_rows=17)
            metadata = current["execution"]
        else:
            current = evaluate_stream(prepared, saved["gamma"], saved["beta"], "gaussian")
            metadata = current["execution"]
        assert execution._TRACE.get() is outer and outer.requested == "auto"
        assert outer.operations == {} and outer.fallbacks == {}
    assert seen and all(device == "cpu" and dtype == torch.float64 for device, dtype in seen)
    assert metadata["requested_device"] == metadata["device"] == "cpu"
    assert set(metadata["factor_devices"]) == {"cpu"}
    assert metadata["factor_devices"]["cpu"] == len(seen)
    assert metadata["fallbacks"] == {}
