from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Barrier

from fastapi.testclient import TestClient
import pytest

from openecon import file_layout
from openecon.data import DataError
from openecon.server import create_app
from openecon.team_store import TeamError
from openecon.workspace import Workspace
import test_team_server
import test_team_store


@pytest.fixture
def team():
    return test_team_store.team.__wrapped__()


@pytest.fixture
def api():
    yield from test_team_server.api.__wrapped__()


header = test_team_server.header


def folder(identifier="a" * 32, name="Models", parent=None):
    return {"kind": "folder", "id": identifier, "name": name, "parent": parent}


def script(identifier="analysis", name="analysis.py", parent=None):
    return {"kind": "script", "id": identifier, "name": name, "parent": parent}


def layout(*rows, version=0):
    return {"version": version, "entries": list(rows)}


@pytest.mark.parametrize("name", ["../bad", "a/b", "a\\b", "a:b", "a?b", "a*b", 'a"b',
                                      "a|b", "<bad>", ".hidden", ".", "..", "trailing.",
                                      " space", "space ", "CON", "NUL.txt", "Lpt9.txt",
                                      "a\x00b", "a\nb", "a\x7fb", "a\u202eb", "a\u2028b",
                                      "é" * 91, "", "\ud800"])
def test_unsafe_names_are_rejected_for_inert_folders(name):
    with pytest.raises(file_layout.FileLayoutError):
        file_layout.validate(layout(folder(name=name)))


def test_names_normalize_nfc_and_allow_embedded_dots():
    result = file_layout.validate(layout(folder(name="Re\u0301sults.v1")))
    assert result["entries"][0]["name"] == "Résults.v1"


@pytest.mark.parametrize("extension", ["md", "tex"])
def test_document_aliases_validate_and_main_analysis_keeps_python(extension):
    named = script("b" * 32, f"report.{extension}")
    assert file_layout.validate(layout(script(), named))["entries"][-1] == named
    with pytest.raises(file_layout.FileLayoutError):
        file_layout.validate(layout(script(name=f"analysis.{extension}")))


@pytest.mark.parametrize("extension", ["md", "tex"])
def test_cross_type_named_alias_preserves_source_versions_and_survives_restart(tmp_path, extension):
    store = Workspace(tmp_path)
    named = store.create_console_script("original.py", "unchanged source")
    original = store.get_file_layout()
    original["entries"][1]["name"] = f"notes.{extension}"
    saved = store.put_file_layout(original)
    reopened = Workspace(tmp_path)
    assert reopened.get_file_layout() == saved
    assert reopened.named_console_script(named["id"]) == named
    assert file_layout.source_name(saved, named["id"], named["name"]) == f"notes.{extension}"
    assert file_layout.source_name(saved, "c" * 32, "fallback.md") == "fallback.md"
    saved["entries"][1]["name"] = "model.py"
    assert reopened.put_file_layout(saved)["entries"][1]["name"] == "model.py"
    assert reopened.named_console_script(named["id"]) == named


@pytest.mark.parametrize("value", [True, -1, "0", 1.1, 2**63 - 1])
def test_version_is_a_bounded_strict_integer(value):
    with pytest.raises(file_layout.FileLayoutError):
        file_layout.validate(layout(version=value))


@pytest.mark.parametrize("rows", [
    [folder(), folder()],
    [{**folder(), "kind": "other"}],
    [{**folder(), "id": "not-a-folder-id"}],
    [{**folder(), "parent": "b" * 32}],
    [folder(parent="a" * 32)],
    [folder(parent="b" * 32), folder("b" * 32, "Second", "a" * 32)],
    [folder(name="Straße"), folder("b" * 32, "STRASSE")],
    [folder(name="analysis.py"), script()],
    [{**script(), "name": "analysis.txt"}],
    [{**script(), "name": "bad .py"}],
    [{**folder(), "path": "elsewhere"}],
])
def test_duplicate_names_unknown_fields_ids_and_cycles_are_rejected(rows):
    with pytest.raises(file_layout.FileLayoutError):
        file_layout.validate(layout(*rows))


def test_identical_names_in_different_folders_and_typed_ids_are_valid():
    identifier = "a" * 32
    rows = [folder(identifier, "Models"), script(identifier, "test.py", identifier),
            {"kind": "dataset", "id": identifier, "name": "test.csv", "parent": identifier},
            folder("b" * 32, "Other"), script("c" * 32, "test.py", "b" * 32)]
    assert file_layout.validate(layout(*rows))["entries"] == rows


def test_depth_folder_count_and_entry_count_are_bounded():
    folders = [folder(f"{i:032x}", f"Level {i}", None if i == 0 else f"{i-1:032x}") for i in range(16)]
    assert len(file_layout.validate(layout(*folders, script(parent=f"{15:032x}")))["entries"]) == 17
    with pytest.raises(file_layout.FileLayoutError):
        file_layout.validate(layout(*folders, folder(f"{16:032x}", "Level 16", f"{15:032x}"), script(parent=f"{16:032x}")))
    with pytest.raises(file_layout.FileLayoutError):
        file_layout.validate(layout(*[folder(f"{i:032x}", f"Folder {i}") for i in range(201)]))
    with pytest.raises(file_layout.FileLayoutError):
        file_layout.validate(layout(*[script(f"{i:032x}", f"file{i}.py") for i in range(2001)]))


def test_defaults_keep_order_and_append_collision_safe_new_files():
    stored = layout(folder(name="data.csv"), script(name="renamed.py", parent="a" * 32), version=5)
    current = [script(), {"kind": "dataset", "id": "file1", "name": "data.csv", "parent": None},
               {"kind": "dataset", "id": "file2", "name": "DATA.CSV", "parent": None}]
    result = file_layout.reconcile(stored, current)
    assert result == layout(*stored["entries"], {**current[1], "name": "data (2).csv"},
                           {**current[2], "name": "DATA (3).CSV"}, version=5)
    assert stored["entries"][1]["name"] == "renamed.py"


def test_local_names_folders_and_reordering_persist_without_code_changes(tmp_path):
    store = Workspace(tmp_path)
    store.save_console_script("print('exact')\n")
    second = store.create_console_script("second.py", "x = 1\n")
    before = {path.name: path.read_bytes() for path in store.console_path.glob("*.json")}
    original = store.get_file_layout()
    assert [row["id"] for row in original["entries"]] == ["analysis", second["id"]]
    proposed = layout(folder(), {**original["entries"][1], "parent": "a" * 32, "name": "first.py"},
                      {**original["entries"][0], "name": "renamed.py"})
    saved = store.put_file_layout(proposed)
    assert saved == {**proposed, "version": 1}
    assert Workspace(tmp_path).get_file_layout() == saved
    assert all((store.console_path / name).read_bytes() == value for name, value in before.items())
    assert store.named_console_script(second["id"])["name"] == "second.py"
    assert store.console_script() == {"name": "analysis.py", "code": "print('exact')\n"}


def test_local_dataset_rename_preserves_extension_bytes_and_provenance(tmp_path):
    source = tmp_path / "actual.csv"
    source.write_text("x,y\n1,2\n2,4\n3,6\n")
    store = Workspace(tmp_path / "project")
    profile = store.import_file(source)
    original_bytes = {path.name: path.read_bytes() for path in store.data_path.iterdir()}
    document = store.get_file_layout()
    renamed = deepcopy(document)
    renamed["entries"][1]["name"] = "better.csv"
    assert store.put_file_layout(renamed)["entries"][1]["name"] == "better.csv"
    assert store.get_dataset(profile["id"]) == profile
    assert all((store.data_path / name).read_bytes() == value for name, value in original_bytes.items())
    renamed["version"] = 1
    renamed["entries"][1]["name"] = "better.parquet"
    with pytest.raises(DataError, match="extension"):
        store.put_file_layout(renamed)


def test_local_compare_and_swap_and_new_file_inventory_conflicts(tmp_path):
    store = Workspace(tmp_path)
    old = store.get_file_layout()
    store.create_console_script()
    with pytest.raises(DataError) as error:
        store.put_file_layout(old)
    assert error.value.code == "FILE_INVENTORY_CONFLICT"
    current = store.get_file_layout()
    saved = store.put_file_layout(current)
    with pytest.raises(DataError) as error:
        Workspace(tmp_path).put_file_layout(current)
    assert error.value.code == "VERSION_CONFLICT"
    assert store.get_file_layout() == saved


def test_two_local_instances_cannot_overwrite_same_version(tmp_path):
    barrier = Barrier(2)
    stores = [Workspace(tmp_path), Workspace(tmp_path)]
    document = stores[0].get_file_layout()
    def save(store):
        barrier.wait()
        try:
            return store.put_file_layout(document)["version"]
        except DataError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(save, stores), key=str) == [1, "VERSION_CONFLICT"]


def test_desktop_trusted_catalog_allows_uncached_scripts(tmp_path):
    store = Workspace(tmp_path)
    known = [{"id": "b" * 32, "name": "remote.py", "version": 8}]
    document = store.get_file_layout(known_scripts=known)
    assert document["entries"][1] == script("b" * 32, "remote.py")
    saved = store.put_file_layout(layout(folder(), script(), script("b" * 32, "local-display.py", "a" * 32)), known_scripts=known)
    assert store.get_file_layout(known_scripts=known) == saved
    assert not (store.console_path / ("b" * 32 + ".json")).exists()
    with pytest.raises(DataError) as error:
        store.put_file_layout({**saved, "version": 1})
    assert error.value.code == "FILE_INVENTORY_CONFLICT"


@pytest.mark.parametrize("known", ["not-a-list", [None], [{"name": "remote.py"}],
                                  [{"id": "../../elsewhere", "name": "remote.py"}],
                                  [{"id": "b" * 32, "name": "../remote.py"}],
                                  [{"id": f"{i:032x}", "name": f"remote{i}.py"} for i in range(201)]])
def test_trusted_catalog_is_still_structurally_validated(tmp_path, known):
    with pytest.raises(DataError):
        Workspace(tmp_path).get_file_layout(known_scripts=known)


@pytest.mark.parametrize("corruption", ["symlink", "directory", "oversize", "json", "shape"])
def test_local_layout_rejects_unsafe_or_corrupt_records(tmp_path, corruption):
    store = Workspace(tmp_path / "project")
    path = store.console_path / "file-layout.json"
    if corruption == "symlink":
        target = tmp_path / "human.json"
        target.write_text(json.dumps(layout(script())))
        path.symlink_to(target)
    elif corruption == "directory":
        path.mkdir()
    else:
        path.write_text(" " * (file_layout.MAX_BYTES + 1) if corruption == "oversize" else
                        "{" if corruption == "json" else '{"version":0,"entries":{}}')
    with pytest.raises(DataError) as error:
        store.get_file_layout()
    assert error.value.code in {"CORRUPT_FILE_LAYOUT", "CORRUPT_RECORD"}


def test_local_endpoint_enforces_session_strict_payload_and_cas(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/files/layout").status_code == 401
        client.headers["X-OpenEcon-Token"] = client.get("/api/session").json()["token"]
        original = client.get("/api/files/layout").json()
        assert original == layout(script())
        saved = client.put("/api/files/layout", json=layout(folder(), script(name="main.py", parent="a" * 32)))
        assert saved.status_code == 200, saved.text
        assert client.put("/api/files/layout", json=original).status_code == 409
        for document in [{**original, "version": True}, {**original, "other": 1}, layout(script(name="../bad.py"))]:
            assert client.put("/api/files/layout", json=document).status_code == 422


def test_team_layout_persists_atomically_and_keeps_canonical_drafts(team):
    second = team.store.create_script(team.pid, team.owner, "second.py", "y=1")
    old_draft = team.store.named_script(team.pid, team.owner, second["id"])
    team.store.add_file(team.pid, team.owner, {"id": "file1", "name": "data.csv", "size_bytes": 5})
    original = team.store.get_file_layout(team.pid, team.viewer)
    proposed = layout(folder(), {**original["entries"][2], "name": "renamed.csv", "parent": "a" * 32},
                      {**original["entries"][1], "name": "renamed.py", "parent": "a" * 32}, script(name="main.py"))
    saved = team.store.put_file_layout(team.pid, team.editor, proposed)
    assert saved == {**proposed, "version": 1}
    assert team.store.get_file_layout(team.pid, team.viewer) == saved
    assert team.store.named_script(team.pid, team.owner, second["id"]) == old_draft
    assert team.store.project(team.pid, team.owner)["files"][0]["name"] == "data.csv"
    with pytest.raises(TeamError) as error:
        team.store.put_file_layout(team.pid, team.owner, proposed)
    assert error.value.code == "VERSION_CONFLICT"


@pytest.mark.parametrize("user", ["viewer", "outsider"])
def test_team_layout_writes_require_project_editor(team, user):
    original = team.store.get_file_layout(team.pid, team.owner)
    with pytest.raises(TeamError) as error:
        team.store.put_file_layout(team.pid, getattr(team, user), original)
    assert error.value.code == ("ROLE_REQUIRED" if user == "viewer" else "NOT_FOUND")
    assert team.store.get_file_layout(team.pid, team.owner) == original


@pytest.mark.parametrize("role", [None, "viewer"])
def test_team_layout_rechecks_membership_inside_save_transaction(team, role):
    original = team.store.get_file_layout(team.pid, team.editor)
    atomic = team.db.atomic
    def revoke_then_save(fn):
        team.db.atomic = atomic
        team.store.change_member(team.pid, team.editor.uid, team.owner, role)
        return atomic(fn)
    team.db.atomic = revoke_then_save
    with pytest.raises(TeamError) as error:
        team.store.put_file_layout(team.pid, team.editor, original)
    assert error.value.code == ("NOT_FOUND" if role is None else "ROLE_REQUIRED")
    assert team.store.get_file_layout(team.pid, team.owner) == original


def test_team_endpoint_roles_isolation_and_stale_inventory(api):
    endpoint = api.prefix + "/files/layout"
    original = api.client.get(endpoint, headers=header("viewer")).json()
    assert original == layout(script())
    proposed = layout(folder(), script(name="main.py", parent="a" * 32))
    assert api.client.put(endpoint, headers=header("viewer"), json=proposed).status_code == 403
    assert api.client.get(endpoint, headers=header("outsider")).status_code == 404
    saved = api.client.put(endpoint, headers=header("editor"), json=proposed)
    assert saved.status_code == 200, saved.text
    assert api.client.get(f"/api/projects/{api.other}/workspace/files/layout", headers=header("owner")).json() == original
    assert api.client.put(endpoint, headers=header("owner"), json=proposed).status_code == 409
    api.store.create_script(api.pid, api.users["owner"], "new.py")
    assert api.client.put(endpoint, headers=header("editor"), json=saved.json()).status_code == 409
    assert api.client.put(endpoint, headers=header("owner"), json={**saved.json(), "version": True}).status_code == 422
    assert api.client.put(endpoint, headers=header("owner"), json={**saved.json(), "extra": 1}).status_code == 422


def test_team_corrupt_layout_is_a_controlled_error(team):
    team.db.put(f"oe_projects/{team.pid}/workspace/file-layout", {"wrong": 1})
    with pytest.raises(TeamError) as error:
        team.store.get_file_layout(team.pid, team.owner)
    assert error.value.code == "CORRUPT_FILE_LAYOUT"


def test_team_failed_transaction_cannot_save_a_partial_layout(team):
    original = team.store.get_file_layout(team.pid, team.owner)
    before = deepcopy(team.db.data)
    put = team.db.put
    def fail_project_write(path, value):
        if path == f"oe_projects/{team.pid}":
            raise RuntimeError("storage unavailable")
        return put(path, value)
    team.db.put = fail_project_write
    with pytest.raises(RuntimeError, match="storage unavailable"):
        team.store.put_file_layout(team.pid, team.editor, layout(folder(), script()))
    assert team.db.data == before
    assert team.store.get_file_layout(team.pid, team.owner) == original


def test_team_get_read_rechecks_removal_before_returning_layout(team):
    get = team.db.get
    def revoke_during_read(path):
        if path.endswith("/workspace/file-layout"):
            team.db.get = get
            team.store.change_member(team.pid, team.editor.uid, team.owner, None)
        return get(path)
    team.db.get = revoke_during_read
    with pytest.raises(TeamError) as error:
        team.store.get_file_layout(team.pid, team.editor)
    assert error.value.code == "NOT_FOUND"
