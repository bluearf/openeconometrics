"""Explicit computational opt-in cannot disable live-memory protections."""
import numpy as np
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.streaming_cox import fit_streaming_cox
from openecon.econometrics.streaming_glmm import fit_streaming_glmm
from openecon.econometrics.streaming_work import work_budget
from openecon.resources import use_workspace_budget

from test_econ_streaming_cox import data as cox_data, spec as cox_spec
from test_econ_streaming_glmm import data as panel_data, spec as panel_spec


@pytest.mark.parametrize("value",["0","-1","1.5","true","","１２","nan"])
def test_work_budget_rejects_malformed_or_nonpositive_values(value,monkeypatch):
    monkeypatch.setenv("OPENECON_TEST_WORK_LIMIT",value)
    with pytest.raises(AnalysisError) as error:
        work_budget("OPENECON_TEST_WORK_LIMIT",50)
    assert error.value.code == "invalid_resource_budget"


def test_default_and_explicit_positive_budget(monkeypatch):
    monkeypatch.delenv("OPENECON_TEST_WORK_LIMIT",raising=False)
    assert work_budget("OPENECON_TEST_WORK_LIMIT",50) == 50
    monkeypatch.setenv("OPENECON_TEST_WORK_LIMIT"," 1000000000 ")
    assert work_budget("OPENECON_TEST_WORK_LIMIT",50) == 1_000_000_000


@pytest.mark.parametrize("mode,setting,code",[("exactp","OPENECON_COX_EXACT_WORK_LIMIT","exact_too_large"),
                                             ("tvc","OPENECON_COX_TVC_WORK_LIMIT","tvc_work_limit"),
                                             ("panel","OPENECON_PANEL_LOGIT_WORK_LIMIT","conditional_work_limit")])
def test_real_model_can_raise_only_computational_limit_while_ram_guard_remains(mode,setting,code,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    if mode == "panel":
        frame = panel_data(n_groups=6,size=6)
        frame.y = np.arange(len(frame))%2
        requested,fit = panel_spec("xtlogit",model="fe"),fit_streaming_glmm
    else:
        frame = cox_data(n=65)
        requested,fit = cox_spec("exactp" if mode == "exactp" else "breslow"),fit_streaming_cox
        if mode == "tvc":
            requested.predictors = ["x"]
            requested.columns["tvc"] = ["z"]
    source = Dataset.from_frame(frame)
    monkeypatch.setenv(setting,"1")
    with pytest.raises(AnalysisError) as error:
        fit(requested,source,batch_rows=7)
    assert error.value.code == code
    monkeypatch.setenv(setting,"1000000000")
    result = fit(requested,source,batch_rows=7)
    record = result.extra["conditional_recursion"] if mode == "panel" else result.extra["replay_event_kernel"]
    field = "work_budget" if mode == "panel" else "exact_work_budget" if mode == "exactp" else "tvc_work_budget"
    assert record[field] == 1_000_000_000
    with use_workspace_budget(8),pytest.raises(AnalysisError) as error:
        fit(requested,source,batch_rows=7)
    assert error.value.code == "workspace_limit"
    assert not list(tmp_path.iterdir())
