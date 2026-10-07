"""Large graph transport stays bounded, durable and isolated per workspace."""
from copy import deepcopy
import hashlib
import json
import os
import time

from fastapi.testclient import TestClient
import pytest

from openecon.console_worker import _execute, validate_result, WorkerProtocolError
from openecon.data import DataError
from openecon.plot_artifacts import checked_plot_path, collect_plot_garbage, history_reference, store_plot
from openecon.server import create_app
from openecon.workspace import Workspace
from openecon_charts import PlotSpec


def graph_plot(count=1000):
    graph = {"nodes": [{"id": i, "label": ("Düğüm " + str(i)) * 35, "degree": 2, "group": i % 3}
                       for i in range(count)],
             "edges": [{"source": i, "target": (i + 1) % count, "weight": 1} for i in range(count)],
             "directed": True, "node_count": count, "edge_count": count,
             "shown_node_count": count, "shown_edge_count": count, "sampled": False,
             "selection": "All nodes and edges"}
    return PlotSpec("network", "Araştırma ağı", "", "", [], count, count, 0,
                    {"network": graph, "options": {}})


def test_large_plot_never_inflates_the_worker_pipe_and_reopens_exactly(tmp_path):
    plot = graph_plot()
    record = _execute("print('before')\ndisplay(p)\nprint('after')", {"p": plot}, "large-plot",
                      artifact_workspace=tmp_path)
    assert record["status"] == "ok"
    assert record["events"] == [{"type": "stdout", "text": "before\n"},
                                {"type": "output", "index": 0},
                                {"type": "stdout", "text": "after\n"}]
    validate_result({"kind": "result", "id": "large-plot", "result": record}, "large-plot")
    data = record["outputs"][0]["data"]
    assert data["sample_n"] == plot.sample_n and "artifact" in data
    assert len(json.dumps(record).encode()) < 16000
    ref = data["artifact"]
    assert json.loads(checked_plot_path(tmp_path, ref).read_text()) == plot.model_dump()
    assert store_plot(tmp_path, plot.model_dump()) == data
    assert len(list((tmp_path / "console/network-plots").glob("*.json"))) == 1
    workspace = Workspace(tmp_path)
    workspace.append_console_history(record)
    reopened = Workspace(tmp_path).console_history()
    assert history_reference(reopened, ref["id"]) == ref
    assert json.loads(checked_plot_path(tmp_path, ref).read_text()) == plot.model_dump()
    assert r"\toprule" in record["outputs"][0]["latex"]


def test_corruption_links_cross_project_and_unrecorded_artifacts_are_refused(tmp_path):
    ref = store_plot(tmp_path / "first", graph_plot().model_dump())["artifact"]
    with pytest.raises(DataError, match="unavailable"):
        checked_plot_path(tmp_path / "second", ref)
    with pytest.raises(DataError, match="history"):
        history_reference([], ref["id"])
    with pytest.raises(DataError):
        history_reference([], "../history")
    file = checked_plot_path(tmp_path / "first", ref)
    original = file.read_bytes()
    file.write_bytes(original.replace(b"group", b"grOup", 1))
    with pytest.raises(DataError, match="checksum"):
        checked_plot_path(tmp_path / "first", ref)
    file.unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(original)
    file.symlink_to(outside)
    with pytest.raises(DataError):
        checked_plot_path(tmp_path / "first", ref)


def test_authenticated_endpoint_only_reads_recorded_plot_from_current_workspace(tmp_path):
    store = Workspace(tmp_path)
    descriptor = store_plot(tmp_path, graph_plot().model_dump())
    ref = descriptor["artifact"]
    with TestClient(create_app(tmp_path)) as client:
        url = "/api/console/plots/" + ref["id"]
        assert client.get(url).status_code == 401
        client.headers["X-OpenEcon-Token"] = client.get("/api/session").json()["token"]
        assert client.get(url).status_code == 404
        record = _execute("p", {"p": graph_plot()}, "large-plot", artifact_workspace=tmp_path)
        store.append_console_history(record)
        response = client.get(url)
        assert response.status_code == 200
        assert response.json() == graph_plot().model_dump()
        assert response.headers["cache-control"] == "no-store"
        assert client.get(url, headers={"Origin": "https://other.example"}).status_code == 403


@pytest.mark.parametrize("mutation", [
    {"id": "../../secret"}, {"bytes": 128 * 1024 * 1024 + 1}, {"bytes": True},
    {"version": True}, {"version": 2}, {"extra": "url"},
])
def test_worker_rejects_malformed_reference_metadata(tmp_path, mutation):
    record = _execute("p", {"p": graph_plot()}, "large-plot", artifact_workspace=tmp_path)
    copied = deepcopy(record)
    copied["outputs"][0]["data"]["artifact"].update(mutation)
    with pytest.raises(WorkerProtocolError, match="reference"):
        validate_result({"kind": "result", "id": "large-plot", "result": copied}, "large-plot")


def test_plot_cap_failure_does_not_leave_partial_files(tmp_path, monkeypatch):
    import openecon.plot_artifacts as module
    monkeypatch.setattr(module, "MAX_PLOT_BYTES", 100)
    with pytest.raises(DataError, match="128 MiB"):
        store_plot(tmp_path, graph_plot().model_dump())
    assert not list((tmp_path / "console/network-plots").iterdir())


def test_store_cap_never_evicts_another_recorded_plot(tmp_path, monkeypatch):
    import openecon.plot_artifacts as module
    first = store_plot(tmp_path, graph_plot().model_dump())
    monkeypatch.setattr(module, "MAX_STORE_BYTES", first["artifact"]["bytes"])
    other = graph_plot(501)
    with pytest.raises(DataError, match="1 GiB"):
        store_plot(tmp_path, other.model_dump())
    assert checked_plot_path(tmp_path, first["artifact"]).is_file()
    assert len(list((tmp_path / "console/network-plots").glob("*.json"))) == 1


def test_gc_only_deletes_expired_verified_cache_entries_absent_from_project_history(tmp_path):
    first = store_plot(tmp_path, graph_plot().model_dump())
    second = store_plot(tmp_path, graph_plot(501).model_dump())
    first_path = checked_plot_path(tmp_path, first["artifact"])
    second_path = checked_plot_path(tmp_path, second["artifact"])
    timestamp = time.time()
    for path in (first_path, second_path):
        os.utime(path, (timestamp - 1000, timestamp - 1000))
    folder = first_path.parent
    user = folder / "notes.json"
    user.write_text('{"user": "keep"}')
    unrelated = b'{"kind":"user-data","values":[1,2,3]}'
    foreign = folder / (hashlib.sha256(unrelated).hexdigest() + ".json")
    foreign.write_bytes(unrelated)
    os.utime(foreign, (timestamp - 1000, timestamp - 1000))
    corrupt = folder / ("a" * 64 + ".json")
    corrupt.write_text('{"kind":"network","not":"an owned checksum"}')
    os.utime(corrupt, (timestamp - 1000, timestamp - 1000))
    outside = tmp_path / "user-outside.json"
    outside.write_bytes(first_path.read_bytes())
    link = folder / ("b" * 64 + ".json")
    link.symlink_to(outside)
    history = [{"outputs": [{"type": "plot", "data": first}]}]
    result = collect_plot_garbage(tmp_path, history=history, now=timestamp)
    assert result["deleted_files"] == 1 and result["deleted_bytes"] == second["artifact"]["bytes"]
    assert result["retained_referenced"] == 1 and result["retained_unverified"] == 2
    assert first_path.is_file() and not second_path.exists()
    assert user.read_text() == '{"user": "keep"}' and foreign.read_bytes() == unrelated
    assert corrupt.is_file() and link.is_symlink() and outside.is_file()


def test_gc_grace_preserves_just_created_plot_before_execution_is_recorded(tmp_path):
    descriptor = store_plot(tmp_path, graph_plot().model_dump())
    path = checked_plot_path(tmp_path, descriptor["artifact"])
    result = collect_plot_garbage(tmp_path, history=[])
    assert result["retained_young"] == 1 and path.exists()
    expired = collect_plot_garbage(tmp_path, history=[], now=path.stat().st_mtime + 301)
    assert expired["deleted_files"] == 1 and not path.exists()


def test_gc_releases_orphaned_storage_before_cap_check_and_preserves_current_history(tmp_path, monkeypatch):
    import openecon.plot_artifacts as module
    first = store_plot(tmp_path, graph_plot().model_dump())
    path = checked_plot_path(tmp_path, first["artifact"])
    os.utime(path, (time.time() - 1000, time.time() - 1000))
    monkeypatch.setattr(module, "MAX_STORE_BYTES", first["artifact"]["bytes"])
    next_plot = store_plot(tmp_path, graph_plot(501).model_dump())
    assert not path.exists() and checked_plot_path(tmp_path, next_plot["artifact"]).exists()
    recorded = checked_plot_path(tmp_path, next_plot["artifact"])
    os.utime(recorded, (time.time() - 1000, time.time() - 1000))
    (tmp_path / "console/history.json").write_text(json.dumps({"history": [{"outputs": [{"type": "plot", "data": next_plot}]}]}))
    with pytest.raises(DataError, match="1 GiB"):
        store_plot(tmp_path, graph_plot().model_dump())
    assert recorded.exists()


@pytest.mark.parametrize("history", [None, {}, [None], [{"outputs": None}], [{"outputs": [None]}],
                                     [{"outputs": [{"type": "plot", "data": {"artifact": {"id": "bad"}}}]}]])
def test_gc_preserves_files_if_recorded_history_cannot_be_verified(tmp_path, history):
    descriptor = store_plot(tmp_path, graph_plot().model_dump())
    path = checked_plot_path(tmp_path, descriptor["artifact"])
    os.utime(path, (time.time() - 1000, time.time() - 1000))
    history_path = tmp_path / "console/history.json"
    history_path.write_text("not json" if history is None else json.dumps({"history": history}))
    with pytest.raises(DataError, match="preserved"):
        collect_plot_garbage(tmp_path)
    assert path.exists()


def test_gc_rejects_linked_or_unbounded_history_and_directory_without_touching_user_files(tmp_path, monkeypatch):
    import openecon.plot_artifacts as module
    descriptor = store_plot(tmp_path, graph_plot().model_dump())
    path = checked_plot_path(tmp_path, descriptor["artifact"])
    os.utime(path, (time.time() - 1000, time.time() - 1000))
    history = tmp_path / "console/history.json"
    outside = tmp_path / "other-project-history.json"
    outside.write_text('{"history": []}')
    history.symlink_to(outside)
    with pytest.raises(DataError, match="Linked"):
        collect_plot_garbage(tmp_path)
    history.unlink()
    history.write_text('{"history": []}')
    monkeypatch.setattr(module, "MAX_HISTORY_BYTES", 1)
    with pytest.raises(DataError, match="preserved"):
        collect_plot_garbage(tmp_path)
    assert path.exists() and outside.read_text() == '{"history": []}'


def test_gc_empty_workspace_and_invalid_options_are_safe(tmp_path):
    assert collect_plot_garbage(tmp_path)["deleted_files"] == 0
    for options in ({"grace_seconds": -1}, {"grace_seconds": True}, {"grace_seconds": float("nan")},
                    {"now": float("inf")}, {"now": True}):
        with pytest.raises(DataError):
            collect_plot_garbage(tmp_path, **options)


def test_platform_without_directory_fd_support_preserves_cache_and_still_stores_plots(tmp_path, monkeypatch):
    descriptor = store_plot(tmp_path, graph_plot().model_dump())
    path = checked_plot_path(tmp_path, descriptor["artifact"])
    os.utime(path, (time.time() - 1000, time.time() - 1000))
    monkeypatch.setattr(os, "supports_fd", set())
    result = collect_plot_garbage(tmp_path, history=[])
    assert not result["cleanup_supported"] and result["deleted_files"] == 0 and path.exists()
    second = store_plot(tmp_path, graph_plot(501).model_dump())
    assert checked_plot_path(tmp_path, second["artifact"]).exists()


def test_gc_recognizes_canonical_managed_header_even_when_caller_kind_key_is_last(tmp_path):
    payload = graph_plot().model_dump()
    payload = {**{key: value for key, value in payload.items() if key != "kind"}, "kind": "network"}
    descriptor = store_plot(tmp_path, payload)
    path = checked_plot_path(tmp_path, descriptor["artifact"])
    assert json.loads(path.read_text()) == payload
    os.utime(path, (time.time() - 1000, time.time() - 1000))
    assert collect_plot_garbage(tmp_path, history=[])["deleted_files"] == 1
    assert not path.exists()


def test_history_expiration_collects_old_graph_without_touching_current_graph(tmp_path):
    workspace = Workspace(tmp_path)
    old = store_plot(tmp_path, graph_plot().model_dump())
    current = store_plot(tmp_path, graph_plot(501).model_dump())
    old_path = checked_plot_path(tmp_path, old["artifact"])
    os.utime(old_path, (time.time() - 1000, time.time() - 1000))
    history = [{"outputs": [{"type": "plot", "data": old}]}] + [{"outputs": []} for _ in range(499)]
    (workspace.console_path / "history.json").write_text(json.dumps({"history": history}))
    workspace.append_console_history({"outputs": [{"type": "plot", "data": current}]})
    assert not old_path.exists()
    assert checked_plot_path(tmp_path, current["artifact"]).exists()
