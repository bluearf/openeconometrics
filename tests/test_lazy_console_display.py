"""Dataset display and variables never enumerate a full replayable source."""
import pandas as pd

from openecon.console_worker import _execute, _variables, validate_result
from openecon.dataset import Dataset


def test_unknown_rows_display_is_a_bounded_latex_preview():
    seen = []
    def factory():
        for start in range(0, 5000, 50):
            seen.append(start)
            yield pd.DataFrame({"x": range(start, start + 50)})
    source = Dataset.from_batches(factory, ["x"])
    assert _variables({"df": source})[0]["preview"] == "unknown rows × 1 columns · batch source"
    assert not seen
    result = _execute("display(df)", {"df": source}, "lazy-display")
    assert result["status"] == "ok", result["error"]
    validate_result({"kind": "result", "id": "lazy-display", "result": result}, "lazy-display")
    table = result["outputs"][0]
    assert table["type"] == "table" and len(table["data"]["rows"]) == 50
    assert table["data"]["total_rows_known"] is False
    assert "total row count was not computed" in table["latex"]
    assert seen == [0]


def test_known_rows_display_preserves_global_shape_and_reports_first_rows():
    source = Dataset.from_frame(pd.DataFrame({"x": range(5000)}))
    result = _execute("display(df)", {"df": source}, "known-display")
    assert result["status"] == "ok", result["error"]
    data = result["outputs"][0]["data"]
    assert data["total_rows"] == 5000 and len(data["rows"]) == 50
    assert "total_rows_known" not in data
    assert _variables({"df": source})[0]["preview"].startswith("5,000 rows")
