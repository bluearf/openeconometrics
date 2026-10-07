"""Real process execution, recovery, durable source, and bounded display output."""
from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from openecon.console import ConsoleError, ConsoleSession
from openecon.plotting import hist, scatter
from openecon.server import create_app
from openecon.workspace import Workspace


@pytest.fixture
def console(tmp_path):
    session = ConsoleSession(Workspace(tmp_path / "workspace"))
    yield session
    session.close()


def test_real_python_preserves_variables_and_last_expression(console):
    first = console.execute("values = [i * i for i in range(5)]\nprint('computed', len(values))\nvalues")
    assert first["status"] == "ok"
    assert first["stdout"] == "computed 5\n"
    assert len(first["outputs"]) == 1
    assert first["outputs"][0]["type"] == "text"
    assert first["outputs"][0]["data"] == "[0, 1, 4, 9, 16]"
    assert "[0, 1, 4, 9, 16]" in first["outputs"][0]["latex"]
    second = console.execute("sum(values)")
    assert second["outputs"][0]["data"] == "30"
    assert any(item["name"] == "values" for item in second["variables"])
    assert first["session_generation"] == second["session_generation"]
    saved = Workspace(console.workspace.path).console_history()
    assert saved[0]["code"] == first["code"]
    assert saved[-1]["outputs"] == second["outputs"]


def test_python_errors_are_visible_and_session_recovers(console):
    syntax = console.execute("for:")
    assert syntax["status"] == "error"
    assert syntax["error"]["type"] == "SyntaxError"
    runtime = console.execute("preserved = 10\n1 / 0")
    assert runtime["error"]["type"] == "ZeroDivisionError"
    assert "ZeroDivisionError" in runtime["error"]["traceback"]
    assert console.execute("preserved + 1")["outputs"][0]["data"] == "11"


def test_text_table_model_and_chart_keep_execution_order_in_saved_history(console):
    result = console.execute("""print('before')
df = oe.example()
display(df.head(2))
print('between')
model = oe.ols(data=df, y='wage', x=['education', 'experience'])
display(model)
display(oe.plot.coefficients(model))
print('hello')
print('hello2')
""")
    assert result['status'] == 'ok', result['error']
    assert [item['type'] for item in result['outputs']] == ['table', 'model', 'plot']
    assert result['events'] == [
        {'type': 'stdout', 'text': 'before\n'},
        {'type': 'output', 'index': 0},
        {'type': 'stdout', 'text': 'between\n'},
        {'type': 'output', 'index': 1},
        {'type': 'output', 'index': 2},
        {'type': 'stdout', 'text': 'hello\nhello2\n'},
    ]
    assert result['stdout'] == 'before\nbetween\nhello\nhello2\n'
    saved = Workspace(console.workspace.path).console_history()[-1]
    assert saved['events'] == result['events']
    assert console.snapshot()['history'][-1]['events'] == result['events']


def test_output_timeline_preserves_partial_outputs_and_stderr_before_error(console):
    result = console.execute("import sys\ndisplay('first')\nprint('last', file=sys.stderr)\n1 / 0")
    assert result['status'] == 'error'
    assert result['events'] == [
        {'type': 'output', 'index': 0}, {'type': 'stdout', 'text': 'last\n'},
    ]
    assert result['error']['type'] == 'ZeroDivisionError'


def test_display_tables_plots_and_model(console):
    result = console.execute("""df = oe.example()
display(df.head(3))
model = oe.ols(data=df, y='wage', x=['education', 'experience'])
display(model)
display(oe.plot.coefficients(model))
oe.plot.scatter(data=df, x='education', y='wage')
""")
    assert result["status"] == "ok", result["error"]
    assert [item["type"] for item in result["outputs"]] == ["table", "model", "plot", "plot"]
    table = result["outputs"][0]["data"]
    assert table["total_rows"] == 3 and len(table["rows"]) == 3
    assert result["outputs"][1]["data"]["nobs"] == 480
    assert result["outputs"][2]["data"]["kind"] == "coefficients"
    assert result["outputs"][3]["data"]["sample_n"] == 480


def test_independent_chart_package_displays_in_same_console(console):
    result = console.execute("""import openecon_charts as charts
chart = charts.area(data={'period': ['A', 'B', 'C'], 'value': [-2, None, 5]}, x='period', y='value')
display(chart)
oe.plot.bar(data={'group': ['North', 'South'], 'value': [12, -3]}, x='group', y='value')
""")
    assert result["status"] == "ok", result["error"]
    assert [item["type"] for item in result["outputs"]] == ["plot", "plot"]
    area, bar = [item["data"] for item in result["outputs"]]
    assert area["config"]["series"][0]["values"] == [-2, None, 5]
    assert bar["config"]["series"][0]["values"] == [12, -3]
    assert area["kind"] == bar["kind"] == "d3"


def test_load_imported_workspace_dataset(console):
    dataset = console.workspace.create_example()
    result = console.execute(f"df = oe.load_dataset({dataset['id']!r})\nlen(df)")
    assert result["status"] == "ok", result["error"]
    assert result["outputs"][0]["data"] == "480"


def test_stdout_and_display_are_bounded(console):
    result = console.execute("print('ğ' * 100000)\nfor i in range(40): display(i)")
    assert result["status"] == "ok"
    assert len(result["stdout"].encode("utf-8")) < 66000
    assert "truncated" in result["stdout"]
    assert len(result["outputs"]) == 20
    from openecon.output_events import validate_output_events
    validate_output_events(result['events'], result['stdout'], result['outputs'])


def test_unicode_truncation_stays_ordered_across_later_ascii_and_displays(console):
    result = console.execute("print('a' * (64*1024-1), end='')\nprint('ğ', end='')\n"
                             "display('first')\nprint('x', end='')\ndisplay('second')")
    assert result['status'] == 'ok', result['error']
    assert result['events'] == [
        {'type': 'stdout', 'text': result['stdout']},
        {'type': 'output', 'index': 0}, {'type': 'output', 'index': 1},
    ]
    assert result['stdout'].count('[Output truncated') == 1


def test_timeout_stops_process_and_restarts_empty_namespace(console):
    console.execute("kept = 'before timeout'")
    old_pid = console.snapshot()["status"]["pid"]
    result = console.execute("while True: pass", timeout_seconds=.1)
    assert result["status"] == "timeout"
    assert result["state_reset"] is True
    assert console.snapshot()["variables"] == []
    restarted = console.execute("'kept' in globals()")
    assert restarted["outputs"][0]["data"] == "False"
    assert console.snapshot()["status"]["pid"] != old_pid
    assert restarted["session_generation"] > result["session_generation"]


def test_interrupt_and_busy_are_independent_of_running_worker(console):
    console.execute("ready = True")
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(console.execute, "import time\ntime.sleep(30)")
        deadline = time.monotonic() + 5
        while not console.snapshot()["status"]["running"] and time.monotonic() < deadline:
            time.sleep(.01)
        with pytest.raises(ConsoleError, match="already running"):
            console.execute("1+1")
        stopped = console.interrupt()
        assert stopped["status"] == "interrupted"
        result = pending.result(timeout=5)
    assert result["status"] == "interrupted"
    assert console.execute("'ready' in globals()")["outputs"][0]["data"] == "False"


def test_worker_exit_and_reset_are_explicit(console):
    crashed = console.execute("import os\nos._exit(7)")
    assert crashed["status"] == "error"
    assert crashed["error"]["type"] == "WORKER_EXITED"
    console.execute("a = 5")
    history_count = len(console.snapshot()["history"])
    console.reset()
    assert len(console.snapshot()["history"]) == history_count
    assert console.execute("'a' in globals()")["outputs"][0]["data"] == "False"


def test_startup_timeout_is_bounded_and_retryable(tmp_path):
    session = ConsoleSession(Workspace(tmp_path / "startup"), startup_timeout=0)
    try:
        failed = session.execute("1 + 1")
        assert failed["status"] == "error"
        assert failed["error"]["type"] == "WORKER_START_FAILED"
        session.startup_timeout = 30
        assert session.execute("1 + 1")["outputs"][0]["data"] == "2"
    finally:
        session.close()


@pytest.mark.parametrize("exit_mode", ["timeout", "interrupt", "reset", "close", "crash"])
def test_supervisor_reclaims_only_its_spill_directory(console, exit_mode):
    unrelated = console.workspace.path / ".openecon-scratch-user-file"
    unrelated.mkdir()
    sentinel = unrelated / "keep.txt"
    sentinel.write_text("user data")
    ready = console.execute("import os\nfrom pathlib import Path\n"
                            "scratch = Path(os.environ['OPENECON_SCRATCH_DIRECTORY'])\n"
                            "(scratch / 'unfinished.sqlite').write_bytes(b'partial spill')\nstr(scratch)")
    assert ready["status"] == "ok", ready["error"]
    scratch = console._scratch_directory
    assert scratch.parent == console.workspace.path and (scratch / "unfinished.sqlite").is_file()
    if exit_mode == "timeout":
        assert console.execute("while True: pass", timeout_seconds=.1)["status"] == "timeout"
    elif exit_mode == "interrupt":
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(console.execute, "import time\ntime.sleep(30)")
            deadline = time.monotonic() + 5
            while not console.status()["running"] and time.monotonic() < deadline:
                time.sleep(.01)
            console.interrupt()
            assert pending.result(timeout=5)["status"] == "interrupted"
    elif exit_mode == "crash":
        assert console.execute("os._exit(7)")["error"]["type"] == "WORKER_EXITED"
    else:
        getattr(console, exit_mode)()
    assert not scratch.exists()
    assert sentinel.read_text() == "user data"
    if exit_mode != "close":
        assert console.execute("import os\nos.environ['OPENECON_SCRATCH_DIRECTORY']")["status"] == "ok"
        assert console._scratch_directory != scratch


def test_scratch_creation_failure_closes_session_resources(console, monkeypatch):
    def fail(**kwargs):
        raise OSError("scratch unavailable")
    monkeypatch.setattr("openecon.console.tempfile.mkdtemp", fail)
    result = console.execute("1 + 1")
    assert result["status"] == "error"
    assert console._scratch_directory is None
    assert console.status()["pid"] is None


def test_plot_counts_sampling_and_missing_are_truthful():
    frame = pd.DataFrame({"x": list(range(3000)) + [None], "y": list(range(3000)) + [1]})
    plot = scatter(data=frame, x="x", y="y")
    assert (plot.total_n, plot.sample_n, plot.dropped_n) == (3000, 2000, 1)
    histogram = hist(data=frame, x="x", bins=10)
    assert sum(row["count"] for row in histogram.data) == 3000
    with pytest.raises(ValueError, match="bins"):
        hist(data=frame, x="x", bins=0)


def test_table_display_preserves_group_labels_multiindex_and_exact_integers():
    # These values identify groups and observations; replacing them with UI row
    # numbers or rounded JavaScript integers changes the meaning of the output.
    from openecon.console_worker import _execute
    namespace = {"pd": pd}
    grouped = _execute(
        "pd.Series([10., 20.], index=pd.Index(['North', 'South'], name='region'), name='wage')",
        namespace, "groups",
    )
    table = grouped["outputs"][0]["data"]
    assert table["index_names"] == ["region"]
    assert table["index"] == [["North"], ["South"]]
    assert table["rows"] == [[10.], [20.]]
    multiple = _execute(
        "pd.DataFrame({'id': [9007199254740993, 10**100]}, "
        "index=pd.MultiIndex.from_tuples([('North', 2025), ('South', 2026)], names=['region', 'year']))",
        namespace, "multi",
    )
    assert multiple["status"] == "ok", multiple["error"]
    table = multiple["outputs"][0]["data"]
    assert table["index_names"] == ["region", "year"]
    assert table["index"] == [["North", 2025], ["South", 2026]]
    assert table["rows"] == [["9007199254740993"], [str(10**100)]]


def test_console_api_token_script_execution_and_persistence(tmp_path):
    app = create_app(tmp_path / "api")
    with TestClient(app) as client:
        assert client.post("/api/console/execute", json={"code": "1+1"}).status_code == 401
        token = client.get("/api/session").json()["token"]
        headers = {"X-OpenEcon-Token": token}
        assert client.get("/api/console", headers=headers).json()["history"] == []
        script = {"code": "x = 41\nx + 1", "name": "analysis.py"}
        assert client.put("/api/console/script", headers=headers, json=script).json() == script
        assert client.get("/api/console/script", headers=headers).json() == script
        # Saving and inspecting a script do not execute it.
        assert client.get("/api/console", headers=headers).json()["variables"] == []
        execution = client.post("/api/console/execute", headers=headers, json={"code": script["code"]}).json()
        assert execution["outputs"][0]["data"] == "42"
        assert client.get("/api/console", headers=headers).json()["history"][-1]["id"] == execution["id"]
        assert client.post("/api/console/reset", headers=headers).json()["status"] == "reset"
        assert client.get("/api/console", headers=headers).json()["variables"] == []
        assert client.put("/api/console/script", headers=headers, json={"code": "", "name": "../bad.py"}).status_code == 422


def test_identified_python_runs_after_document_alias_conversion_and_terminal_stays_compatible(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    calls = []
    monkeypatch.setattr(app.state.console, "execute", lambda code, **kwargs: calls.append(code) or {"code": code})
    with TestClient(app) as client:
        client.headers["X-OpenEcon-Token"] = client.get("/api/session").json()["token"]
        created = client.post("/api/console/scripts", json={"name": "draft.md", "code": "1 + 2"}).json()
        layout = client.get("/api/files/layout").json()
        layout["entries"][1]["name"] = "model.py"
        assert client.put("/api/files/layout", json=layout).status_code == 200
        identified = client.post("/api/console/execute", json={"code": "1 + 2", "script_id": created["id"]})
        assert identified.status_code == 200 and calls == ["1 + 2"]
        terminal = client.post("/api/console/execute", json={"code": "2 + 3"})
        assert terminal.status_code == 200 and calls == ["1 + 2", "2 + 3"]
        assert client.post("/api/console/execute", json={"code": "3 + 4", "script_id": "analysis"}).status_code == 200
        assert calls == ["1 + 2", "2 + 3", "3 + 4"]


def _fail_worker_start(connection, workspace, *_options):
    from openecon.console_worker import send_json
    send_json(connection, {"kind": "ready", "ready": False, "error": "ImportError: test startup failure"})
    connection.close()


def test_worker_import_failure_is_returned_without_hanging(tmp_path, monkeypatch):
    import openecon.console_worker as worker
    original = worker.worker_main
    monkeypatch.setattr(worker, "worker_main", _fail_worker_start)
    session = ConsoleSession(Workspace(tmp_path / "import-failure"))
    try:
        failed = session.execute("1 + 1")
        assert failed["error"]["type"] == "WORKER_START_FAILED"
        assert "ImportError" in failed["error"]["message"]
        assert session.snapshot()["status"]["running"] is False
        monkeypatch.setattr(worker, "worker_main", original)
        assert session.execute("1 + 1")["outputs"][0]["data"] == "2"
    finally:
        session.close()


@pytest.mark.skipif(__import__("os").name != "posix", reason="Process group lifecycle is POSIX-specific")
def test_timeout_stops_inherited_subprocesses(console):
    sentinel = console.workspace.path / "should-not-exist.txt"
    child_code = f"import time; from pathlib import Path; time.sleep(1); Path({str(sentinel)!r}).write_text('orphan')"
    code = f"import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', {child_code!r}])\nwhile True: pass"
    result = console.execute(code, timeout_seconds=.15)
    assert result["status"] == "timeout"
    time.sleep(1.1)
    assert not sentinel.exists()


def test_cost_flow_summary_has_table_and_publication_output(console):
    record = console.execute("""import openecon as oe
graph = oe.network([{'source': 's', 'target': 't', 'capacity': 2}], directed=True, weight='capacity')
flow = graph.min_cost_flow({'s': -1, 't': 1}, {('s', 't'): 3})
display(flow)
display(flow['flows'])
""")
    assert record['status'] == 'ok', record.get('error')
    assert [item['type'] for item in record['outputs']] == ['table', 'table']
    assert all('tabular' in item['latex'] for item in record['outputs'])
