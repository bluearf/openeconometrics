"""Endpoint/sample contracts fail before estimation and survive JSON round trips."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival_ext import common


def test_mixed_endpoints_and_canonical_json():
    data = common.interval_data([2, 0, 3, 4], [2, 1, 6, np.inf])
    assert data.kind == ["exact", "left", "interval", "right"]
    assert data.settings["upper"] == [2, 1, 6, None]
    assert json.loads(json.dumps(data.settings, allow_nan=False)) == data.settings
    assert data.lo.dtype == data.hi.dtype == torch.float64
    assert data.lo.device.type == data.hi.device.type == "cpu"


@pytest.mark.parametrize("lower,upper", [
    ([0],[None]), ([0],[0]), ([2],[1]), ([-1],[2]), ([1],[float("nan")]),
    ([1],[float("-inf")]), ([1],[True]), ([True],[2]), ([1],[]),
    ([1,2],[3]), ([1e-13],[2]), ([1],[1e13]), ([10**1000],[None]), ([1],[10**1000]),
])
def test_invalid_endpoint_sets(lower, upper):
    with pytest.raises(AnalysisError):
        common.interval_data(lower, upper)


@pytest.mark.parametrize("time,event", [
    ([0],[1]), ([1],[True]), ([1],[-1]), ([1],[1.5]), ([1],[None]),
    ([1],[np.nan]), ([1],[2**31]), ([1,2],[1]), ([1],[10**1000]),
])
def test_event_coding_cannot_silently_recode(time,event):
    with pytest.raises(AnalysisError):
        common.event_data(time,event)


@pytest.mark.parametrize("values", [np.ones((2,1)), pd.DataFrame({"x":[1,2]}), [[1],[2]], "time", torch.ones(2,requires_grad=True)])
def test_positional_complete_vector_domain(values):
    with pytest.raises(AnalysisError):
        common.interval_data(values,[2,3])


def test_event_rows_preserve_original_positions_and_integer_causes():
    data = common.event_data([3,1,2,1],[2,1,0,4])
    assert data.settings["time"] == [3,1,2,1]
    assert data.settings["event"] == [2,1,0,4]
    assert data.event.dtype == torch.int64


@pytest.mark.parametrize("controls", [{"device":"cuda"},{"device":"mps"},{"weights":[1,1]}])
def test_unsupported_controls(controls):
    with pytest.raises(AnalysisError):
        common.interval_data([1,2],[2,3],**controls)
    with pytest.raises(AnalysisError):
        common.event_data([1,2],[1,0],**controls)


def test_input_and_joint_workspace_preflight(monkeypatch):
    with pytest.raises(AnalysisError):
        common.interval_data([1]*4097,[2]*4097)
    with pytest.raises(AnalysisError):
        common.workspace(10,257)
    with pytest.raises(AnalysisError):
        common.workspace(10,2,support=257)
    monkeypatch.setattr(common,"workspace_budget_bytes",lambda:1024)
    with pytest.raises(AnalysisError):
        common.workspace(100,10)


@pytest.mark.parametrize("times", [[1,1],[2,1],[-1],[float("inf")],[],[True]])
def test_requested_times_are_complete_and_prespecified(times):
    with pytest.raises(AnalysisError):
        common.prediction_times(times)


@pytest.mark.parametrize("level", [0,1,-1,float("nan"),True,".95"])
def test_invalid_confidence(level):
    with pytest.raises(AnalysisError):
        common.confidence(level)


def test_global_torch_defaults_do_not_change_cpu_precision():
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        data = common.interval_data([1,2],[2,None])
        events = common.event_data([1,2],[1,0])
        times = common.prediction_times([0,1])
        assert all(t.device.type == "cpu" for t in (data.lo,data.hi,events.t,events.event,times))
        assert common.confidence(.95)[1] == pytest.approx(1.959963984540054)
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(old)


def test_complete_metadata_and_result_tables():
    data = common.event_data([1,2],[1,0])
    result = common.output("example",{"estimate":[{"time":1,"value":.5}]},data.settings,["iid"])
    restored = {row.setting:json.loads(row.json) for row in result["settings"].itertuples()}
    assert restored["time"] == [1,2]
    assert restored["complete_inputs_saved"]
    assert restored["notes"] == ["iid"]
    with pytest.raises(AnalysisError):
        common.output("bad",{},dict(data.settings,value=np.nan))
