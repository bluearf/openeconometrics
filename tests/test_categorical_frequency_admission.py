"""Count validation must precede filtering and numerical allocation."""

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical.frequency import BYTES, WORK, _atom, _sample


def admit(data, **kwargs):
    options = dict(missing="drop", max_bytes=BYTES, max_work=WORK)
    options.update(kwargs)
    return _sample(data, ["x"], "f", operation="admission test", **options)


def sample():
    return pd.DataFrame({"x": range(6), "f": [1, 2, 3, 4, 5, 6]}, index=["dup"]*6)


@pytest.mark.parametrize("bad", [True, -1, 1.5, float("inf"), "3", 10**100, [2], complex(2, 0)])
def test_invalid_counts_not_hidden_by_missing_feature_or_tensor_conversion(bad, monkeypatch):
    data = sample().astype({"f": object, "x": object})
    data.iloc[0, 0] = None
    data.iat[0, 1] = bad
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("allocated before invalid count rejection"))
    with pytest.raises(AnalysisError):
        admit(data)


def test_physical_zero_missing_count_and_feature_alignment():
    data = sample().astype({"x": object, "f": object})
    data.iat[0, 1] = 0
    data.iat[0, 0] = ["ignored zero-weight object"]
    data.iat[1, 1] = pd.NA
    got = admit(data)
    assert got["positions"] == [2, 3, 4, 5]
    assert got["counts"] == [3, 4, 5, 6]
    assert got["zero_positions"] == [0]
    assert got["missing_positions"] == [1]
    assert got["frequency_total"] == 18
    assert got["frame"].x.tolist() == [2, 3, 4, 5]
    with pytest.raises(AnalysisError):
        admit(data, missing="raise")


def test_integer_float_counts_and_large_total_remain_physical():
    data = sample()
    data["f"] = [100_000_000.0]*6
    got = admit(data)
    assert len(got["frame"]) == 6 and got["frequency_total"] == 600_000_000
    assert all(type(x) is int for x in got["counts"])
    data["f"] = [200_000_000]*6
    with pytest.raises(AnalysisError):
        admit(data)


def test_counts_validated_even_when_feature_missing_would_remove_excess_total():
    data = sample().astype({"x": float})
    data.loc[:, "f"] = [1_000_000_000, 1, 1, 1, 1, 1]
    data.iloc[0, 0] = np.nan
    with pytest.raises(AnalysisError):
        admit(data)


@pytest.mark.parametrize("options", [{"max_bytes": 1}, {"max_work": 1}, {"max_bytes": BYTES+1}, {"max_work": WORK+1}])
def test_admission_precedes_values_and_tensors(options, monkeypatch):
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("allocated before admission"))
    with pytest.raises(AnalysisError):
        admit(sample(), **options)


def test_duplicate_columns_and_oversize_physical_input_refused():
    data = sample()
    with pytest.raises(AnalysisError):
        admit(pd.concat([data, data[["f"]]], axis=1))
    with pytest.raises(AnalysisError):
        admit(pd.DataFrame({"x": range(3001), "f": np.ones(3001)}))


@pytest.mark.parametrize("bad", [np.array([1, 2]), np.array([1]), [1], {"x": 1}, "x"*257, 2**53, float("nan")])
def test_category_labels_fail_closed_on_nonscalar_or_unbounded_values(bad):
    with pytest.raises(AnalysisError):
        _atom(bad)


def test_wide_input_projects_before_row_copies(monkeypatch):
    data = pd.DataFrame(np.zeros((8, 1000)), columns=[f"unused{i}" for i in range(1000)])
    data["x"], data["f"] = range(8), 1
    row_copy_widths = []
    original = pd.DataFrame.take

    def observed(self, indices, axis=0, **kwargs):
        if axis == 0:
            row_copy_widths.append(len(self.columns))
        return original(self, indices, axis=axis, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "take", observed)
    result = admit(data, max_bytes=8192)
    assert result["workspace"]["estimated_workspace_bytes"] == 6144
    assert row_copy_widths and set(row_copy_widths) == {1}
