"""NARDL exact sample ordering, global partial sums and constrained QR."""
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.streaming_nardl import fit_streaming_nardl
from test_streaming_ardl import data as ardl_data, source
from test_streaming_var import numeric_tree, numerical_blocks


@pytest.fixture(scope="module")
def data():
    return ardl_data.__wrapped__()


@pytest.mark.parametrize("block", [3, 651])
@pytest.mark.parametrize("options", [{"lags": [2, 1]}, {"lags": [2, 1], "ec": False},
                                    {"lags": [2, 1], "covariance": "robust"},
                                    {"lags": [2, 1], "covariance": "HC3", "long_run_symmetric": ["x"]},
                                    {"lags": [2, 2, 1], "short_run_symmetric": ["x"]},
                                    {"lags": [2, 2], "long_run_symmetric": ["x"], "short_run_symmetric": ["x"]},
                                    {"maxlags": [3, 2]},
                                    {"maxlags": [3, 2], "ic": "bic", "long_run_symmetric": ["x"]},
                                    {"lags": [2, 1], "trend": "trend", "restricted": True, "exog": ["z"]}])
def test_complete_nardl_contract(data, block, options, monkeypatch):
    dense = oe.nardl(data=data, y="y", x=["x"], time="t", **options)
    numerical_blocks(monkeypatch, block)
    result = fit_streaming_nardl(dense.spec, source(data, block))
    assert result.nobs == dense.nobs and result.dropped_rows == dense.dropped_rows
    assert [c.term for c in result.coefficients] == [c.term for c in dense.coefficients]
    numeric_tree([c.estimate for c in dense.coefficients], [c.estimate for c in result.coefficients], 3e-7)
    numeric_tree(dense.covariance_matrix, result.covariance_matrix, 3e-7)
    numeric_tree(dense.metrics, result.metrics, 3e-7)
    numeric_tree(dense.tests, result.tests, 3e-7)
    numeric_tree(dense.extra, result.extra, 3e-7)
    numeric_tree(oe.nardl_multipliers(dense, steps=5).to_dict(),
                 oe.nardl_multipliers(result, steps=5).to_dict(), 3e-7)


def test_partial_sum_origin_and_declared_datetime_grid(data, tmp_path):
    frame = data.copy()
    frame.t = pd.date_range("2020-01-01", periods=len(frame), freq="h", tz="Europe/Istanbul")
    frame.loc[[0, len(frame)-1], "y"] = float("nan")
    frame = frame.sample(frac=1, random_state=67).reset_index(drop=True)
    path = tmp_path/"nardl.parquet"
    frame.to_parquet(path, index=False, row_group_size=11)
    dense = oe.nardl(data=frame, y="y", x=["x"], time="t", time_delta="1h", missing="drop", lags=[2, 1])
    result = fit_streaming_nardl(dense.spec, oe.scan(path))
    numeric_tree(dense.metrics, result.metrics, 3e-7)
    assert result.extra["partial_sum_origin"]["row"] == dense.extra["partial_sum_origin"]["row"]
    wrong = dense.spec.model_copy(update={"options": {**dense.spec.options, "time_delta": "2h"}})
    with pytest.raises(AnalysisError, match="consecutive"):
        fit_streaming_nardl(wrong, oe.scan(path))


def test_one_sided_changes_cannot_identify_two_effects(data):
    dense = oe.nardl(data=data, y="y", x=["x"], time="t", lags=[2, 1])
    monotone = data.assign(x=list(range(len(data))))
    with pytest.raises(AnalysisError, match="positive and negative"):
        fit_streaming_nardl(dense.spec, source(monotone, 1))
