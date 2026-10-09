"""Calendar/release refusals and complete state are independent of tensor solvers."""
import json
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.temporal.common import prepare, output


def fixture():
    return dict(low=[10.,12.],indicator=list(range(1,9)),low_periods=["2020","2021"],
                high_periods=[f"{year}Q{q}" for year in (2020,2021) for q in range(1,5)])


@pytest.mark.parametrize("aggregation,expected",[("sum",[10.,26.]),("mean",[2.5,6.5]),("first",[1.,5.]),("last",[4.,8.])])
def test_explicit_aggregation_matrix(aggregation,expected):
    p=prepare(**fixture(),aggregation=aggregation)
    assert (p.C@p.x[:,0]).tolist()==expected
    assert p.settings["indicator"]==[[float(i)] for i in range(1,9)]


@pytest.mark.parametrize("changes,code",[
    ({"low_periods":["2021","2020"]},"invalid_calendar"),
    ({"low_periods":["2020","2020"]},"invalid_calendar"),
    ({"high_periods":["2020Q1","2020Q2","2020Q3","2020Q4","2021Q1","2021Q2","2021Q4","2022Q1"]},"invalid_calendar"),
    ({"high_periods":[f"{y}Q{q}" for y in (2019,2020) for q in range(1,5)]},"invalid_calendar"),
    ({"low_frequency":"Q"},"invalid_calendar"),
    ({"indicator":[1.]*7},"invalid_input"),
    ({"indicator":[float("nan")]*8},"missing_values"),
    ({"indicator":[True]*8},"invalid_input"),
    ({"low":[[1.],[2.]]},"invalid_input"),
    ({"indicator":[[1,2],[3],*[ [4,5] for _ in range(6)]]},"invalid_input"),
    ({"device":"cuda"},"unsupported_device"),
    ({"weights":[1,1]},"unsupported_weights"),
    ({"aggregation":"average"},"invalid_option"),
    ({"as_of":"2022-01-01"},"invalid_release"),
    ({"indicator":[1e60]*8},"invalid_input"),
    ({"indicator":(x for x in range(8))},"unsupported_input"),
])
def test_refusal_no_silent_calendar_or_sample_repair(changes,code):
    with pytest.raises(AnalysisError) as exc:
        prepare(**(fixture()|changes))
    assert exc.value.code==code


def test_complete_release_boundary_no_future_rows_or_timezone_normalization():
    base=fixture()|dict(as_of="2022-01-01",low_releases=["2021-01-10","2022-01-01"],high_releases=["2022-01-01"]*8)
    p=prepare(**base)
    assert p.settings["low_releases"][1]==p.settings["as_of"]
    with pytest.raises(AnalysisError,match="released after"):
        prepare(**(base|dict(high_releases=["2022-01-02"]*8)))
    with pytest.raises(AnalysisError,match="timezone"):
        prepare(**(base|dict(as_of="2022-01-01Z".replace("01Z","01T00:00:00Z"))))
    with pytest.raises(AnalysisError,match="match"):
        prepare(**(base|dict(low_releases=["2021-01-01"])))
    with pytest.raises(AnalysisError,match="calendar ends"):
        prepare(**(base|dict(as_of="2021-06-01",low_releases=["2021-01-01"]*2,high_releases=["2021-01-01"]*8)))


@pytest.mark.parametrize("low_frequency,low,high",[
    ("Y",["2020","2021"],[f"{y}-{m:02d}" for y in (2020,2021) for m in range(1,13)]),
    ("Q",["2020Q4","2021Q1"],["2020-10","2020-11","2020-12","2021-01","2021-02","2021-03"]),
])
def test_cross_year_month_routes(low_frequency,low,high):
    p=prepare([3.,6.],[1.]*len(high),low_periods=low,high_periods=high,low_frequency=low_frequency,high_frequency="M")
    assert (p.C@torch.ones(len(high),dtype=torch.float64)).tolist()==[len(high)/2]*2


def test_complete_settings_json_and_latex_state():
    p=prepare(**fixture())
    values=torch.tensor([2.5]*4+[3.]*4,dtype=torch.float64)
    r=output("example",p,values)
    saved=json.loads(json.dumps(r.attrs,allow_nan=False))
    assert saved["low_periods"]==["2020","2021"]
    assert len(r["series"])==8 and len(r["aggregation"])==2
    assert "\\begin{tabular}" in r.to_latex()


def test_workspace_refused_before_tensor_allocation(monkeypatch):
    monkeypatch.setattr("openecon.econometrics.temporal.common.workspace_budget_bytes",lambda:1)
    with pytest.raises(AnalysisError,match="workspace"):
        prepare(**fixture())


def test_torch_default_device_and_dtype_do_not_change_public_precision():
    with torch.device("meta"):
        p=prepare(**fixture())
    assert p.y.device.type==p.x.device.type==p.C.device.type=="cpu"
    assert p.y.dtype==p.x.dtype==p.C.dtype==torch.float64
