"""Named table axes survive finite v1 summary persistence without label coercion."""
import json

import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.summary_state import restore_summary, summary_state


def test_named_numeric_confusion_axes_and_full_json_roundtrip():
    frame = table([[3, 1], [0, 4]], index=[0, 1], columns=[0, 1])
    frame.index.name, frame.columns.name = "actual", "predicted"
    output = TableSet({"classification_table": frame}, procedure="named_test", n=8)
    encoded = summary_state(output)
    restored = restore_summary(encoded)
    pd.testing.assert_frame_equal(restored["classification_table"], frame)
    assert list(map(type, restored["classification_table"].index)) == [int, int]
    assert list(map(type, restored["classification_table"].columns)) == [int, int]
    assert summary_state(restored) == encoded


def test_named_mixed_scalar_axes_retain_types_and_duplicates():
    frame = table([[.1, .2], [.3, .4], [.5, .6]],
                  index=["row", 9, "row"], columns=["probability", 7])
    frame.index.name, frame.columns.name = 13, "measure"
    restored = restore_summary(summary_state(TableSet({"mixed": frame})))
    pd.testing.assert_frame_equal(restored["mixed"], frame)
    assert restored["mixed"].index.tolist() == ["row", 9, "row"]
    assert restored["mixed"].columns.tolist() == ["probability", 7]
    assert type(restored["mixed"].index[1]) is int
    assert type(restored["mixed"].columns[1]) is int


def test_legacy_v1_without_axis_names_remains_readable():
    legacy = {"schema": "openecon.summary.v1", "title": "legacy", "attrs": {"n": 2},
              "tables": {"values": {"columns": ["x"], "index": [8, 9], "data": [[1], [2]]}}}
    output = restore_summary(json.dumps(legacy))
    pd.testing.assert_frame_equal(output["values"], table([[1], [2]], columns=["x"], index=[8, 9]))
    assert output.title == "legacy" and output.attrs == {"n": 2}


@pytest.mark.parametrize("field,value", [
    ("index_names", "actual"), ("index_names", []), ("index_names", ["a", "b"]),
    ("column_names", [{"name": "invalid"}]), ("column_names", [["nested"]]),
    ("column_names", [float("inf")]),
])
def test_malformed_optional_axis_names_refuse(field, value):
    output = TableSet({"values": table([[1]], columns=["x"], index=[9])})
    payload = json.loads(summary_state(output))
    payload["tables"]["values"][field] = value
    with pytest.raises(AnalysisError) as error:
        restore_summary(json.dumps(payload))
    assert error.value.code == "invalid_state"
