"""Public first-stage assumptions, unsupported requests and resource boundaries."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.control_function import KINDS


def frame(n=120):
    generator = torch.Generator(device="cpu").manual_seed(73)
    values = torch.randn((n, 5), generator=generator, dtype=torch.float64)
    w, z, u, error, other = values.T
    d = 0.4*w+0.9*z+u
    y = 0.3+0.7*w+0.8*d+0.4*u+error
    return pd.DataFrame(dict(w=w, z=z, d=d, y=y, other=other))


def test_registry_capabilities_and_lazy_public_surface_do_not_load_tensor_runtime():
    code = "import json,sys,openecon as oe; c=oe.capabilities(); print(json.dumps([c['control_function'], 'torch' in sys.modules]))"
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    contract, loaded = json.loads(output)
    assert not loaded
    assert contract["public_api"] == [*KINDS, "cf_predict", "cf_restore"]
    assert contract["stata_parity_validated"] is False
    assert contract["budgets"]["resident_state_rows"] == 5000
    assert contract["budgets"]["Dataset_rows"] == "resource/work budget; no fixed row cap"


@pytest.mark.parametrize("options", [
    {"instruments": []}, {"instruments": ["w"]}, {"x": ["d"]},
    {"covariance": "nonrobust"}, {"covariance": "HC1"},
    {"device": "cuda"}, {"max_iterations": 201}, {"tolerance": 0},
    {"max_work": 1}, {"covariance": "cluster", "cluster": ["z", "other"]},
])
def test_public_refusal_does_not_mutate_data(options):
    data = frame()
    before = data.copy(deep=True)
    kwargs = dict(y="y", endogenous="d", x=["w"], instruments=["z"])
    kwargs.update(options)
    with pytest.raises((AnalysisError, ValueError)):
        oe.cfregress(data=data, **kwargs)
    pd.testing.assert_frame_equal(data, before)


def test_binary_first_stage_refuses_and_large_original_sample_replays():
    data = frame()
    data["d"] = (data.d > 0).astype(int)
    with pytest.raises(AnalysisError, match="Binary endogenous"):
        oe.cfregress(data=data, y="y", endogenous="d", x=["w"], instruments=["z"])
    result = oe.cfregress(data=frame(5001), y="y", endogenous="d", x=["w"], instruments=["z"])
    assert result.nobs == 5001
    assert result.provenance["streaming"]["dense_observation_matrix"] is False
    assert result.extra["control_function_state"]["schema"] == "openecon.control_function.stream.v2"


def test_explicit_cpu_float64_preserves_ambient_dtype_rng_and_device():
    data = frame()
    rng = torch.get_rng_state().clone()
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            result = oe.cfregress(data=data, y="y", endogenous="d", x=["w"], instruments=["z"])
            assert torch.empty(0).device.type == "meta"
        assert torch.get_default_dtype() == torch.float32
        assert result.nobs == len(data)
        assert result.inference["small_sample_correction"] == 1
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        torch.set_default_dtype(old)


def test_saved_result_contains_no_fabricated_likelihood_or_weak_iv_guarantee():
    result = oe.cfregress(data=frame(), y="y", endogenous="d", x=["w"], instruments=["z"])
    assert result.metrics.get("log_likelihood") is None
    assert result.metrics.get("aic") is None and result.metrics.get("bic") is None
    assert result.tests["control_coefficient_zero"]["distribution"] == "chi2"
    state = result.extra["control_function_state"]
    assert result.provenance["control_function_state_sha256"] == state["integrity_sha256"]
    assert "control_function" in oe.capabilities()["families"]


@pytest.mark.parametrize("use_dataset", [False, True])
def test_public_stream_fit_reports_real_cpu_trace_under_ambient_auto(use_dataset, monkeypatch):
    from openecon.engines import execution

    data = frame(121)
    if use_dataset:
        data = oe.Dataset.from_frame(data)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    native = execution._preferred_device

    def cpu_only(trace, block):
        assert trace.requested == "cpu"
        return native(trace, block)

    monkeypatch.setattr(execution, "_preferred_device", cpu_only)
    with execution.execution_scope("auto") as outer:
        result = oe.cfregress(data=data, y="y", endogenous="d", x=["w"],
                              instruments=["z"], batch_rows=43)
        assert execution._TRACE.get() is outer and outer.operations == {}
    receipt = result.provenance["execution"]
    assert receipt["requested_device"] == receipt["device"] == "cpu"
    assert receipt["factor_devices"]["cpu"] > 0
    assert set(receipt["factor_devices"]) == {"cpu"}


def test_streamed_missing_warning_covers_rows_after_reporting_preview():
    data = frame(601)
    data.loc[500, "d"] = float("nan")
    result = oe.cfregress(data=oe.Dataset.from_frame(data), y="y", endogenous="d", x=["w"],
                          instruments=["z"], batch_rows=43, missing="drop")
    assert result.nobs == 600 and result.dropped_rows == 1
    assert "Excluded 1 observation(s) with missing model inputs." in result.warnings


def test_catalog_codec_restores_every_existing_field_and_reserves_marker():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("cf_editor_codec", root/"scripts/generate_editor_api.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    entry = dict(name="openecon.cfregress", kind="function", signature="cfregress(*, x=None)", description="Complete.", parameters=[])
    stored = generator.encode_catalog([entry])
    assert stored[0]["S"] == "(*, x=None)"
    assert generator.decode_catalog(stored) == [entry]
    assert generator.decode_catalog([{**stored[0], "signature": "legacy()"}])[0]["signature"] == "legacy()"
    with pytest.raises(ValueError, match="reserved compact parenthesis"):
        generator.encode_catalog([{**entry, "signature": "(broken)"}])
