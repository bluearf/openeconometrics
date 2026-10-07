"""Resident chart metadata survives publication validation and console restart."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openecon.console import ConsoleSession
from openecon.team_output import validate_plot
from openecon.workspace import Workspace
import openecon_charts as charts


def plot():
    frame = pd.DataFrame({"x": np.arange(5005, dtype=float), "y": np.ones(5005)})
    frame.loc[1, "x"] = -100.
    return charts.scatter(data=frame, x="x", y="y", color="#123456").model_dump()


def test_publication_keeps_processing_counts_full_extents_and_options():
    value = plot()
    before = deepcopy(value)
    assert validate_plot(value) == value == before
    assert value["config"]["processing"]["extents"]["x"] == [-100., 5004.]
    assert min(row["x"] for row in value["data"]) > -100.
    assert len(json.dumps(value)) < 180000


@pytest.mark.parametrize("field,value", [("source_rows", 0), ("source_rows", 5004),
    ("passes", True), ("reader_batch_rows", 65537), ("projected_working_limit_bytes", 2**30),
    ("projected_columns", ["x", "x"]), ("sample_limit", 2001),
    ("extents", {"x": [0., 5004.], "y": [2., 3.]}), ("extents", {"x": [0., 5004.]}),
    ("mode", "unknown"), ("unknown", "unused")])
def test_publication_rejects_inconsistent_or_malformed_processing(field, value):
    data = plot()
    data["config"]["processing"][field] = value
    with pytest.raises(ValueError):
        validate_plot(data)


def test_million_row_code_panel_example_and_saved_restart(tmp_path):
    workspace = Workspace(tmp_path / "owned-resident-charts")
    source = (Path(__file__).resolve().parents[1] / "docs/examples/resident_charts.py").read_text()
    session = ConsoleSession(workspace)
    try:
        result = session.execute(source, timeout_seconds=60)
        assert result["status"] == "ok", result.get("error")
        assert [item["type"] for item in result["outputs"]] == ["table", "plot", "plot", "plot"]
        assert "RESIDENT_CHARTS_RECEIPT:" in result["stdout"]
        for output in result["outputs"][1:]:
            assert validate_plot(output["data"]) == output["data"]
            assert "\\begin{tikzpicture}" in output["latex"]
        saved = Workspace(workspace.path).console_history()[-1]
        assert saved["outputs"] == result["outputs"] and saved["events"] == result["events"]
        session.close()
        session = ConsoleSession(Workspace(workspace.path))
        assert session.snapshot()["history"][-1]["outputs"] == result["outputs"]
    finally:
        session.close()
