"""The published example runs in the code panel and retains its CLI export path."""
import json
from pathlib import Path
import runpy

import pytest

from openecon.console import ConsoleSession
from openecon.workspace import Workspace
from openecon_charts.timeline import unpack


EXAMPLE = Path(__file__).resolve().parents[1] / "docs/examples/network_workbench.py"
LAYOUTS = ["d3-force", "forceatlas2", "circular", "grid", "radial", "hierarchical",
           "geographic", "community", "fixed"]


@pytest.mark.parametrize("argv", [[], ["openecon-runtime", "--port", "0", "--data-root", "unused"]])
def test_entire_example_displays_in_real_panel_worker_without_cli_arguments_or_exports(tmp_path, argv):
    workspace = Workspace(tmp_path / "panel")
    session = ConsoleSession(workspace)
    try:
        assert session.execute(f"import sys\nsys.argv = {argv!r}")["status"] == "ok"
        result = session.execute(EXAMPLE.read_text())
        assert result["status"] == "ok", result["error"]
        assert result["error"] is None
        assert [item["type"] for item in result["outputs"]] == ["table", "table"] + ["plot"] * 10
        plots = [unpack(item["data"]) for item in result["outputs"] if item["type"] == "plot"]
        assert [plot["config"]["options"]["layout"] for plot in plots[:9]] == LAYOUTS
        assert [frame["label"] for frame in plots[-1]["config"]["network"]["frames"]] == ["before", "after"]
        assert result["stdout"] == "9 layouts · 2 snapshots\n"
        assert Workspace(workspace.path).console_history()[-1]["outputs"] == result["outputs"]
        assert not list(workspace.path.rglob("*.html"))
        assert not list(workspace.path.rglob("layout-*.json"))
        assert not list(workspace.path.rglob("manifest.json"))
    finally:
        session.close()


def test_cli_entry_keeps_all_export_files_and_saved_view_readback(tmp_path, capsys):
    module = runpy.run_path(str(EXAMPLE), run_name="network_workbench_cli_test")
    output = tmp_path / "exports"
    assert module["main"](["--output", str(output)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest == json.loads((output / "manifest.json").read_text())
    assert manifest["layouts"] == LAYOUTS and manifest["snapshots"] == 2
    assert len(manifest["files"]) == 25
    assert manifest["full_view_roundtrip"] and manifest["typed_identity_roundtrip"]
    assert all((output / name).is_file() for name in manifest["files"])
