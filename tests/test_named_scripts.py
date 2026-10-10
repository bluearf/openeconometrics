"""Named drafts stay inert, durable, versioned and scoped to their project."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from openecon.data import DataError
from openecon.desktop_runtime import create_desktop_app
from openecon.server import create_app
from openecon.team_auth import TeamAuth, TeamIdentity
from openecon.team_server import create_team_app
from openecon.team_storage import MemoryStorage
from openecon.team_store import MemoryDocuments, TeamError, TeamStore
from openecon.workspace import Workspace


ORIGIN = "https://named-scripts.example.com"
FIREBASE_PROJECT = "openecon-script-tests"
PROJECT_A = "a" * 32
PROJECT_B = "b" * 32


def make_team():
    db = MemoryDocuments()
    users = {
        uid: TeamIdentity(uid, f"{uid}@example.com", uid, True)
        for uid in ("owner", "editor", "viewer", "outsider")
    }
    store = TeamStore(db, owner_email=users["owner"].email, public_origin=ORIGIN)
    store.me(users["owner"])
    pid = store.create_project(users["owner"], "Scripts")["id"]
    other = store.create_project(users["owner"], "Private")["id"]
    for uid, role in (("editor", "editor"), ("viewer", "viewer")):
        invitation = store.invite(pid, users["owner"], users[uid].email, role)
        store.accept(invitation["id"], users[uid])
    return SimpleNamespace(db=db, store=store, users=users, pid=pid, other=other)


@pytest.fixture
def team():
    return make_team()


@pytest.fixture(params=("local", "team"))
def drafts(request, tmp_path):
    if request.param == "local":
        workspace = Workspace(tmp_path)
        return SimpleNamespace(
            kind="local",
            store=workspace,
            error=DataError,
            list=workspace.console_scripts,
            read=workspace.named_console_script,
            create=workspace.create_console_script,
            save=workspace.save_named_console_script,
        )
    team = make_team()
    user = team.users["editor"]
    return SimpleNamespace(
        kind="team",
        store=team.store,
        team=team,
        error=TeamError,
        list=lambda: team.store.scripts(team.pid, user),
        read=lambda script_id: team.store.named_script(team.pid, user, script_id),
        create=lambda name="untitled.py", code="", script_id=None: team.store.create_script(
            team.pid, user, name, code, script_id=script_id
        ),
        save=lambda script_id, code, version: team.store.save_named_script(
            team.pid, user, script_id, code, version
        ),
    )


def race(functions):
    barrier = Barrier(len(functions))

    def run(function):
        barrier.wait(timeout=10)
        try:
            return "ok", function()
        except (DataError, TeamError) as error:
            return error.code, error

    with ThreadPoolExecutor(max_workers=len(functions)) as executor:
        return list(executor.map(run, functions))


def test_two_files_keep_independent_versions_and_listing_has_no_code(drafts):
    initial = drafts.list()
    assert initial == {"scripts": [{"id": "analysis", "name": "analysis.py", "version": 0}]}
    first = drafts.create("first.py", "first = 1")
    second = drafts.create("second.py", "second = 2")
    assert set(first) == {"id", "name", "code", "version"}
    assert first["version"] == second["version"] == 0
    assert first["id"] != second["id"]
    updated_first = drafts.save(first["id"], "first = 3", 0)
    updated_second = drafts.save(second["id"], "second = 4", 0)
    assert updated_first == {**first, "code": "first = 3", "version": 1}
    assert updated_second == {**second, "code": "second = 4", "version": 1}
    assert drafts.read("analysis")["code"] is None
    assert drafts.read(first["id"]) == updated_first
    assert drafts.read(second["id"]) == updated_second
    listing = drafts.list()["scripts"]
    assert len(listing) == 3
    assert {item["id"] for item in listing} == {"analysis", first["id"], second["id"]}
    assert all(set(item) == {"id", "name", "version"} for item in listing)


def test_duplicate_names_get_case_insensitive_suffix_without_overwrite(drafts):
    first = drafts.create("script.py", "original")
    second = drafts.create("SCRIPT.py", "second")
    third = drafts.create("script.py", "third")
    assert first["name"] == "script.py"
    assert second["name"] == "SCRIPT_2.py"
    assert third["name"] == "script_3.py"
    assert drafts.read(first["id"])["code"] == "original"
    assert len({item["name"].casefold() for item in drafts.list()["scripts"]}) == 4


@pytest.mark.parametrize("extension,content", [("md", "# Bulgular\n**Ücret** analizi.\n"),
                                              ("tex", r"\begin{tabular}{lr}Ücret & 1.25\\\end{tabular}")])
def test_source_documents_are_inert_versioned_and_keep_their_suffix(drafts, extension, content):
    created = drafts.create(f"Bulgular.{extension}", content)
    duplicate = drafts.create(f"BULGULAR.{extension}", "separate document")
    assert duplicate["name"] == f"BULGULAR_2.{extension}"
    updated = drafts.save(created["id"], content + "\n% Saved text", 0)
    assert updated["version"] == 1
    assert drafts.read(created["id"])["code"] == content + "\n% Saved text"
    assert drafts.read(duplicate["id"])["code"] == "separate document"
    with pytest.raises(drafts.error) as caught:
        drafts.save(created["id"], "late overwrite", 0)
    assert caught.value.code == "VERSION_CONFLICT"
    assert drafts.read(created["id"]) == updated
    assert {row["name"] for row in drafts.list()["scripts"]} == {
        "analysis.py", f"Bulgular.{extension}", f"BULGULAR_2.{extension}"}


@pytest.mark.parametrize("extension", ["md", "tex"])
@pytest.mark.parametrize("unsafe", ["../report.{ext}", ".hidden.{ext}", "CON.{ext}",
                                    "bad .{ext}", "bad..{ext}", "bad\u2028.{ext}",
                                    "bad.{ext} ", "bad.{ext}.py.exe"])
def test_documents_keep_the_same_safe_filename_rules(drafts, extension, unsafe):
    with pytest.raises(drafts.error):
        drafts.create(unsafe.format(ext=extension), "inert")
    assert drafts.list() == {"scripts": [{"id": "analysis", "name": "analysis.py", "version": 0}]}


def test_create_retries_same_id_and_assigned_name_are_idempotent(drafts):
    drafts.create("report.py", "first")
    script_id = uuid4().hex
    created = drafts.create("report.py", "second", script_id=script_id)
    assert created["name"] == "report_2.py"
    assert drafts.create(created["name"], created["code"], script_id=script_id) == created
    with pytest.raises(drafts.error) as caught:
        drafts.create(created["name"], "changed by retry", script_id=script_id)
    assert caught.value.code == "SCRIPT_ID_CONFLICT"
    assert drafts.read(script_id) == created
    assert len(drafts.list()["scripts"]) == 3


def test_parallel_creation_uses_unique_names_and_retry_cannot_duplicate(drafts):
    outcomes = race([lambda: drafts.create("model.py", "same") for _ in range(6)])
    assert all(status == "ok" for status, _ in outcomes)
    created = [value for _, value in outcomes]
    assert {value["name"] for value in created} == {
        "model.py",
        *(f"model_{index}.py" for index in range(2, 7)),
    }
    script_id = uuid4().hex
    retry = race([lambda: drafts.create("retry.py", "same", script_id=script_id) for _ in range(4)])
    assert all(status == "ok" for status, _ in retry)
    assert all(value == retry[0][1] for _, value in retry)
    assert len(drafts.list()["scripts"]) == 8


def test_parallel_write_has_one_winner_and_stale_write_cannot_erase_it(drafts):
    created = drafts.create("shared.py")
    outcomes = race(
        [
            lambda: drafts.save(created["id"], "writer_one = 1", 0),
            lambda: drafts.save(created["id"], "writer_two = 2", 0),
        ]
    )
    assert sorted(status for status, _ in outcomes) == ["VERSION_CONFLICT", "ok"]
    winner = next(value for status, value in outcomes if status == "ok")
    assert drafts.read(created["id"]) == winner
    with pytest.raises(drafts.error) as caught:
        drafts.save(created["id"], "late overwrite", 0)
    assert caught.value.code == "VERSION_CONFLICT"
    assert drafts.read(created["id"]) == winner


@pytest.mark.parametrize(
    "name",
    [
        "",
        "../model.py",
        "dir/model.py",
        "dir\\model.py",
        ".hidden.py",
        "model.py\n",
        "model\x00.py",
        "model\x7f.py",
        "model\x85.py",
        "model\u2028.py",
        " model.py",
        "model.py ",
        "model:one.py",
        "model.txt",
        "a" * 181 + ".py",
    ],
)
def test_unsafe_names_rejected_without_catalog_changes(drafts, name):
    before = drafts.list()
    with pytest.raises(drafts.error):
        drafts.create(name, "should never be persisted")
    assert drafts.list() == before


@pytest.mark.parametrize(
    "script_id",
    [
        "../analysis",
        "analysis.py",
        "a/b",
        "A" * 32,
        "0" * 31,
        "g" * 32,
        "analysis\x00",
        "",
        "a" * 129,
    ],
)
def test_unsafe_ids_cannot_read_write_or_create(drafts, script_id):
    before = drafts.list()
    with pytest.raises(drafts.error):
        drafts.read(script_id)
    with pytest.raises(drafts.error):
        drafts.save(script_id, "overwrite", 0)
    with pytest.raises(drafts.error):
        drafts.create("safe.py", "new", script_id=script_id)
    assert drafts.list() == before


@pytest.mark.parametrize("version", [-1, True, 1.0, "0", None, 2**63 - 1])
def test_save_version_is_strict_and_failed_write_preserves_content(drafts, version):
    created = drafts.create("safe.py", "preserved")
    with pytest.raises(drafts.error):
        drafts.save(created["id"], "changed", version)
    assert drafts.read(created["id"]) == created


def test_character_limit_accepts_boundary_and_rejects_next_character(drafts):
    created = drafts.create("boundary.py", "x" * 64000)
    assert len(created["code"]) == 64000
    with pytest.raises(drafts.error):
        drafts.save(created["id"], "x" * 64001, 0)
    with pytest.raises(drafts.error):
        drafts.create("too-long.py", "x" * 64001)
    assert drafts.read(created["id"]) == created


def test_file_quota_counts_analysis_and_retry_at_limit_stays_idempotent(drafts, monkeypatch):
    from openecon import script_contracts

    monkeypatch.setattr(script_contracts, "MAX_SCRIPTS", 3)
    first = drafts.create("first.py")
    second = drafts.create("second.py")
    assert len(drafts.list()["scripts"]) == 3
    assert drafts.create(first["name"], first["code"], script_id=first["id"]) == first
    with pytest.raises(drafts.error):
        drafts.create("third.py")
    assert drafts.read(second["id"]) == second


def test_aggregate_quota_counts_utf8_and_replacement_not_all_history(drafts, monkeypatch):
    from openecon import script_contracts

    monkeypatch.setattr(script_contracts, "MAX_SCRIPT_BYTES", 16)
    first = drafts.create("first.py", "é" * 4)
    second = drafts.create("second.py", "é" * 4)
    with pytest.raises(drafts.error):
        drafts.save(first["id"], "é" * 5, 0)
    assert drafts.read(first["id"]) == first
    smaller = drafts.save(second["id"], "", 0)
    bigger = drafts.save(first["id"], "é" * 8, 0)
    assert bigger["version"] == smaller["version"] == 1
    assert bigger["code"] == "é" * 8


def test_local_migration_preserves_legacy_shape_and_advances_shared_main_version(tmp_path):
    workspace = Workspace(tmp_path)
    legacy_path = workspace.console_path / "script.json"
    legacy = {"name": "analysis.py", "code": "legacy = 41"}
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    original = legacy_path.read_bytes()
    assert workspace.console_script() == legacy
    assert workspace.named_console_script("analysis") == {
        "id": "analysis",
        **legacy,
        "version": 0,
    }
    assert legacy_path.read_bytes() == original
    other = workspace.create_console_script("other.py", "other = 2")
    assert workspace.save_named_console_script("analysis", "legacy = 42", 0)["version"] == 1
    assert workspace.console_script() == {"name": "analysis.py", "code": "legacy = 42"}
    assert workspace.save_console_script("legacy = 43") == {
        "name": "analysis.py",
        "code": "legacy = 43",
    }
    assert workspace.named_console_script("analysis")["version"] == 2
    with pytest.raises(DataError):
        workspace.save_named_console_script("analysis", "stale = 0", 1)
    reopened = Workspace(tmp_path)
    assert reopened.named_console_script(other["id"]) == other
    assert reopened.named_console_script("analysis")["code"] == "legacy = 43"
    assert reopened.named_console_script("analysis")["version"] == 2


def test_team_migration_preserves_existing_main_version_and_old_metadata(team):
    legacy = {
        "name": "analysis.py",
        "code": "legacy = 41",
        "version": 7,
        "updated_at": "2026-01-01T00:00:00+00:00",
        "updated_by": "owner",
    }
    path = f"oe_projects/{team.pid}/workspace/script"
    team.db.put(path, legacy)
    owner = team.users["owner"]
    assert team.store.named_script(team.pid, owner, "analysis") == {
        "id": "analysis",
        "name": "analysis.py",
        "code": "legacy = 41",
        "version": 7,
    }
    assert team.store.script(team.pid, owner) == legacy
    assert team.db.get(path) == legacy
    updated = team.store.save_named_script(team.pid, owner, "analysis", "legacy = 42", 7)
    assert updated["version"] == 8
    assert team.store.script(team.pid, owner)["code"] == "legacy = 42"
    assert team.store.save_script(team.pid, owner, "legacy = 43", 8)["version"] == 9
    assert team.store.named_script(team.pid, owner, "analysis")["version"] == 9


def test_team_read_documents_are_detached_and_other_project_cannot_find_id(team):
    script = team.store.create_script(team.pid, team.users["editor"], "private.py", "private = 17")
    script["code"] = "mutated response"
    first = team.store.named_script(team.pid, team.users["viewer"], script["id"])
    first["name"] = "changed response.py"
    assert (
        team.store.named_script(team.pid, team.users["owner"], script["id"])["code"]
        == "private = 17"
    )
    with pytest.raises(TeamError) as caught:
        team.store.named_script(team.other, team.users["owner"], script["id"])
    assert caught.value.status_code == 404


@pytest.mark.parametrize("role", [None, "viewer"])
@pytest.mark.parametrize("operation", ["create", "save"])
def test_team_mutations_recheck_role_inside_transaction(team, role, operation):
    script = team.store.create_script(team.pid, team.users["owner"], "safe.py", "preserved")
    original = team.db.atomic
    entered = False

    def revoked_before_transaction(function):
        nonlocal entered
        if not entered:
            entered = True
            team.store.change_member(team.pid, "editor", team.users["owner"], role)
        return original(function)

    team.db.atomic = revoked_before_transaction
    with pytest.raises(TeamError) as caught:
        if operation == "create":
            team.store.create_script(team.pid, team.users["editor"], "forbidden.py")
        else:
            team.store.save_named_script(
                team.pid, team.users["editor"], script["id"], "forbidden", 0
            )
    assert caught.value.status_code == (404 if role is None else 403)
    assert team.store.named_script(team.pid, team.users["owner"], script["id"]) == script
    assert len(team.store.scripts(team.pid, team.users["owner"])["scripts"]) == 2


@pytest.mark.parametrize("read", ["list", "main", "named"])
def test_team_reads_recheck_membership_after_draft_or_index_read(team, read):
    owner, editor = team.users["owner"], team.users["editor"]
    script = team.store.create_script(team.pid, owner, "private.py", "secret = 17")
    team.store.save_script(team.pid, owner, "main_secret = 19", 0)
    target = {
        "list": f"oe_projects/{team.pid}/workspace/scripts",
        "main": f"oe_projects/{team.pid}/workspace/script",
        "named": f"oe_projects/{team.pid}/scripts/{script['id']}",
    }[read]
    original = team.db.get
    removed = False

    def removed_during_read(path):
        nonlocal removed
        value = original(path)
        if path == target and not removed:
            removed = True
            team.store.change_member(team.pid, editor.uid, owner)
        return value

    team.db.get = removed_during_read
    with pytest.raises(TeamError) as caught:
        if read == "list":
            team.store.scripts(team.pid, editor)
        else:
            team.store.named_script(
                team.pid, editor, "analysis" if read == "main" else script["id"]
            )
    assert removed
    assert caught.value.status_code == 404


@pytest.mark.parametrize("operation", ["create", "save"])
def test_team_partial_transaction_failure_rolls_back_document_index_and_audit(team, operation):
    owner = team.users["owner"]
    existing = team.store.create_script(team.pid, owner, "safe.py", "preserved")
    script_id = uuid4().hex if operation == "create" else existing["id"]
    target = f"oe_projects/{team.pid}/scripts/{script_id}"
    before = deepcopy(team.db.data)
    original = team.db.put

    def interrupted_after_document_put(path, value):
        original(path, value)
        if path == target:
            raise RuntimeError("synthetic interrupted transaction")

    team.db.put = interrupted_after_document_put
    with pytest.raises(RuntimeError, match="synthetic interrupted transaction"):
        if operation == "create":
            team.store.create_script(team.pid, owner, "new.py", "new = 1", script_id=script_id)
        else:
            team.store.save_named_script(team.pid, owner, script_id, "overwrite = 1", 0)
    assert team.db.data == before
    assert team.store.named_script(team.pid, owner, existing["id"]) == existing


@pytest.fixture(params=("local", "team"))
def api(request, tmp_path):
    if request.param == "local":
        app = create_app(tmp_path)
        with TestClient(app) as client:
            client.headers["X-OpenEcon-Token"] = client.get("/api/session").json()["token"]
            yield SimpleNamespace(client=client, prefix="/api", app=app, kind="local")
        return
    team = make_team()
    revoked = set()

    def verify(token):
        if token not in team.users or token in revoked:
            raise ValueError("invalid")
        user = team.users[token]
        return {
            "uid": user.uid,
            "sub": user.uid,
            "email": user.email,
            "name": user.name,
            "aud": FIREBASE_PROJECT,
            "iss": f"https://securetoken.google.com/{FIREBASE_PROJECT}",
            "firebase": {"sign_in_provider": "password"},
            "email_verified": True,
        }

    storage = MemoryStorage()
    # Drafts remain inert metadata; the sync backend has no execution backend.
    app = create_team_app(
        store=team.store,
        storage=storage,
        public_origin=ORIGIN,
        auth=TeamAuth(FIREBASE_PROJECT, verifier=verify),
        firebase_config={
            "projectId": FIREBASE_PROJECT,
            "authDomain": f"{FIREBASE_PROJECT}.firebaseapp.com",
        },
    )
    with TestClient(app, base_url=ORIGIN) as client:
        client.headers["Authorization"] = "Bearer editor"
        yield SimpleNamespace(
            client=client,
            prefix=f"/api/projects/{team.pid}/workspace",
            app=app,
            kind="team",
            team=team,
            revoked=revoked,
            storage=storage,
        )


def test_http_create_read_save_and_conflict_are_inert(api, tmp_path):
    sentinel = tmp_path / "must-not-run"
    code = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')"
    path = api.prefix + "/console/scripts"
    created_response = api.client.post(path, json={"name": "new.py", "code": code})
    assert created_response.status_code == 201, created_response.text
    created = created_response.json()
    detail = path + "/" + created["id"]
    assert api.client.get(detail).json() == created
    assert api.client.get(path).headers["Cache-Control"] == "no-store"
    saved = api.client.put(detail, json={"code": "kept = 1", "version": 0})
    assert saved.status_code == 200, saved.text
    assert saved.json() == {**created, "code": "kept = 1", "version": 1}
    conflict = api.client.put(detail, json={"code": "lost = 2", "version": 0})
    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["detail"]["code"] == "VERSION_CONFLICT"
    assert api.client.get(detail).json() == saved.json()
    assert not sentinel.exists()
    if api.kind == "local":
        assert api.app.state.console._process is None
        assert api.app.state.workspace.console_history() == []
    else:
        assert api.storage.objects == {}
        assert api.team.store.project(api.team.pid, api.team.users["owner"])["run_count"] == 0


def retired_team_execution(response):
    """The team sync backend refuses every execution request before reading it."""
    assert response.status_code == 410, response.text
    assert response.json()["detail"]["code"] == "CLOUD_EXECUTION_RETIRED"


@pytest.mark.parametrize("extension", ["md", "tex"])
def test_identified_documents_never_start_python_even_when_contents_are_python(api, extension, tmp_path):
    sentinel = tmp_path / "document-must-not-run"
    code = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')"
    created = api.client.post(api.prefix + "/console/scripts",
                              json={"name": f"report.{extension}", "code": code}).json()
    assert created["name"] == f"report.{extension}"
    result = api.client.post(api.prefix + "/console/execute",
                             json={"code": code, "script_id": created["id"]})
    if api.kind == "team":
        retired_team_execution(result)
    else:
        assert result.status_code == 422, result.text
        assert result.json()["detail"]["code"] == "DOCUMENT_NOT_EXECUTABLE"
    assert not sentinel.exists()
    assert api.client.get(api.prefix + "/console").json()["history"] == []
    if api.kind == "local":
        assert api.app.state.console._process is None
    else:
        assert api.storage.objects == {}
        assert api.team.store.project(api.team.pid, api.team.users["owner"])["run_count"] == 0


@pytest.mark.parametrize("extension", ["md", "tex"])
def test_execution_resolves_cross_type_explorer_rename_and_preserves_source(api, extension):
    path = api.prefix + "/console/scripts"
    created = api.client.post(path, json={"name": "report.py", "code": "1 + 2"}).json()
    layout = api.client.get(api.prefix + "/files/layout").json()
    for entry in layout["entries"]:
        if entry["kind"] == "script" and entry["id"] == created["id"]:
            entry["name"] = f"report.{extension}"
    renamed = api.client.put(api.prefix + "/files/layout", json=layout)
    assert renamed.status_code == 200, renamed.text
    assert api.client.get(path + "/" + created["id"]).json() == created
    response = api.client.post(api.prefix + "/console/execute",
                               json={"code": "1 + 2", "script_id": created["id"]})
    if api.kind == "team":
        retired_team_execution(response)
    else:
        assert response.status_code == 422, response.text
        assert response.json()["detail"]["code"] == "DOCUMENT_NOT_EXECUTABLE"
    if api.kind == "local":
        assert api.app.state.console._process is None
    else:
        assert api.storage.objects == {}


@pytest.mark.parametrize("identifier", ["c" * 32, "../elsewhere", "analysis.md"])
def test_identified_execution_cannot_use_unknown_or_unsafe_file_ids(api, identifier):
    response = api.client.post(api.prefix + "/console/execute", json={"code": "1", "script_id": identifier})
    if api.kind == "team":
        retired_team_execution(response)
    else:
        assert response.status_code == (404 if identifier == "c" * 32 else 422)
    if api.kind == "local":
        assert api.app.state.console._process is None
    else:
        assert api.storage.objects == {}


def test_http_post_defaults_to_blank_and_fixed_id_retry_is_safe(api):
    path = api.prefix + "/console/scripts"
    first = api.client.post(path, json={})
    assert first.status_code == 201, first.text
    assert first.json()["name"] == "untitled.py" and first.json()["code"] == ""
    script_id = uuid4().hex
    body = {"id": script_id, "name": "safe.py", "code": "saved = 1"}
    saved = api.client.post(path, json=body)
    assert saved.status_code == 201, saved.text
    assert api.client.post(path, json=body).json() == saved.json()
    conflict = api.client.post(path, json={**body, "code": "overwrite = 2"})
    assert conflict.status_code == 409, conflict.text
    assert api.client.get(path + "/" + script_id).json() == saved.json()


@pytest.mark.parametrize(
    "body",
    [
        {"code": "bad", "version": True},
        {"code": "bad", "version": "0"},
        {"code": "bad", "version": -1},
        {"code": "bad", "version": 2**63 - 1},
        {"code": "bad"},
        {"code": "bad", "version": 0, "name": "renamed.py"},
        {"code": None, "version": 0},
        {"code": "x" * 64001, "version": 0},
    ],
)
def test_http_strict_save_body_cannot_modify_file(api, body):
    path = api.prefix + "/console/scripts"
    created = api.client.post(path, json={"name": "safe.py", "code": "preserved"}).json()
    detail = path + "/" + created["id"]
    assert api.client.put(detail, json=body).status_code == 422
    assert api.client.get(detail).json() == created


@pytest.mark.parametrize(
    "body",
    [
        {"name": "../unsafe.py"},
        {"name": "unsafe.txt"},
        {"name": "bad\u2028.py"},
        {"id": "analysis"},
        {"id": "A" * 32},
        {"code": None},
        {"name": "safe.py", "execute": True},
        {"name": "safe.py", "version": 0},
    ],
)
def test_http_strict_create_body_cannot_create_file(api, body):
    path = api.prefix + "/console/scripts"
    before = api.client.get(path).json()
    assert api.client.post(path, json=body).status_code == 422
    assert api.client.get(path).json() == before


def test_http_authentication_viewer_isolation_and_revocation(api):
    path = api.prefix + "/console/scripts"
    created = api.client.post(path, json={"name": "private.py", "code": "secret = 17"}).json()
    detail = path + "/" + created["id"]
    if api.kind == "local":
        headers = {"X-OpenEcon-Token": "wrong"}
        assert api.client.get(path, headers=headers).status_code == 401
        assert api.client.get(detail, headers=headers).status_code == 401
        assert api.client.post(path, json={}, headers=headers).status_code == 401
        assert (
            api.client.put(
                detail, json={"code": "overwrite", "version": 0}, headers=headers
            ).status_code
            == 401
        )
        return
    viewer = {"Authorization": "Bearer viewer"}
    assert api.client.get(path, headers=viewer).status_code == 200
    assert api.client.get(detail, headers=viewer).json() == created
    assert api.client.post(path, json={}, headers=viewer).status_code == 403
    assert (
        api.client.put(detail, json={"code": "overwrite", "version": 0}, headers=viewer).status_code
        == 403
    )
    outsider = {"Authorization": "Bearer outsider"}
    assert api.client.get(path, headers=outsider).status_code == 404
    assert api.client.get(detail, headers=outsider).status_code == 404
    other = f"/api/projects/{api.team.other}/workspace/console/scripts/{created['id']}"
    assert api.client.get(other, headers={"Authorization": "Bearer owner"}).status_code == 404
    api.revoked.add("editor")
    assert api.client.get(path).status_code == 401
    assert api.client.get(detail).status_code == 401
    assert api.client.post(path, json={}).status_code == 401
    assert api.client.put(detail, json={"code": "overwrite", "version": 0}).status_code == 401
    assert (
        api.team.store.named_script(api.team.pid, api.team.users["owner"], created["id"]) == created
    )


def test_desktop_project_switch_restart_and_token_rotation_preserve_two_scripts(tmp_path):
    saved = {}
    with TestClient(create_desktop_app(tmp_path)) as client:
        prefix_a = f"/api/desktop/projects/{PROJECT_A}/workspace"
        token_a = client.get(prefix_a + "/session").json()["token"]
        headers_a = {"X-OpenEcon-Token": token_a}
        for name, code in (("first.py", "first = 1"), ("second.py", "second = 2")):
            response = client.post(
                prefix_a + "/console/scripts", headers=headers_a, json={"name": name, "code": code}
            )
            assert response.status_code == 201, response.text
            item = response.json()
            saved[item["id"]] = item
        console = client.app.state.desktop_projects.active_app.state.console
        assert console._process is None
        prefix_b = f"/api/desktop/projects/{PROJECT_B}/workspace"
        headers_b = {"X-OpenEcon-Token": client.get(prefix_b + "/session").json()["token"]}
        assert (
            len(client.get(prefix_b + "/console/scripts", headers=headers_b).json()["scripts"]) == 1
        )
        assert client.get(prefix_a + "/console/scripts", headers=headers_a).status_code == 409
        assert client.get(prefix_b + "/console/scripts", headers=headers_a).status_code == 401
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = {"X-OpenEcon-Token": client.get(prefix_a + "/session").json()["token"]}
        assert headers != headers_a
        for script_id, document in saved.items():
            assert (
                client.get(prefix_a + "/console/scripts/" + script_id, headers=headers).json()
                == document
            )
        assert client.app.state.desktop_projects.active_app.state.console._process is None


@pytest.mark.parametrize("target", ["folder", "document", "index", "journal"])
def test_local_named_script_storage_refuses_symlinks_without_touching_target(tmp_path, target):
    workspace = Workspace(tmp_path / "workspace")
    created = workspace.create_console_script("safe.py", "preserved")
    external = tmp_path / "external.json"
    external.write_text('{"untouched": true}', encoding="utf-8")
    if target == "folder":
        external_folder = tmp_path / "external-console"
        external_folder.mkdir()
        moved = workspace.path / "original-console"
        workspace.console_path.rename(moved)
        workspace.console_path.symlink_to(external_folder, target_is_directory=True)
    else:
        path = (
            workspace.console_path
            / {
                "document": f"{created['id']}.json",
                "index": "scripts.json",
                "journal": "scripts-pending.json",
            }[target]
        )
        path.unlink(missing_ok=True)
        path.symlink_to(external)
    with pytest.raises(DataError):
        workspace.named_console_script(created["id"])
    with pytest.raises(DataError):
        workspace.save_named_console_script(created["id"], "overwrite", 0)
    assert external.read_text(encoding="utf-8") == '{"untouched": true}'


def test_local_pending_transaction_recovers_document_and_index_together(tmp_path, monkeypatch):
    import openecon.workspace as workspace_module

    workspace = Workspace(tmp_path)
    original = workspace_module._atomic_json
    interrupted = False

    def fail_index_once(path, document):
        nonlocal interrupted
        if Path(path).name == "scripts.json" and not interrupted:
            interrupted = True
            raise OSError("synthetic interrupted commit")
        return original(path, document)

    # Initialize migration before fault injection so the interrupted transaction
    # belongs to the new file rather than the legacy main draft.
    workspace.console_scripts()
    monkeypatch.setattr(workspace_module, "_atomic_json", fail_index_once)
    script_id = uuid4().hex
    with pytest.raises(OSError):
        workspace.create_console_script("durable.py", "durable = 17", script_id=script_id)
    monkeypatch.setattr(workspace_module, "_atomic_json", original)
    reopened = Workspace(tmp_path)
    recovered = reopened.named_console_script(script_id)
    assert recovered == {
        "id": script_id,
        "name": "durable.py",
        "code": "durable = 17",
        "version": 0,
    }
    assert {item["id"] for item in reopened.console_scripts()["scripts"]} == {"analysis", script_id}
    assert not (reopened.console_path / "scripts-pending.json").exists()
    assert (
        reopened.create_console_script("durable.py", "durable = 17", script_id=script_id)
        == recovered
    )


def test_desktop_cache_import_keeps_exact_name_id_and_independent_local_versions(tmp_path):
    workspace = Workspace(tmp_path)
    other = workspace.create_console_script("other.py", "other = 1")
    script_id = uuid4().hex
    imported = workspace.cache_console_script(script_id, "cloud.py", "cloud = 17", 0)
    assert imported == {"id": script_id, "name": "cloud.py", "code": "cloud = 17", "version": 0}
    assert workspace.cache_console_script(script_id, "cloud.py", "cloud = 17", 0) == imported
    renamed = workspace.cache_console_script(script_id, "renamed.py", "cloud = 17", 0)
    assert renamed == {**imported, "name": "renamed.py", "version": 1}
    assert workspace.cache_console_script(script_id, "renamed.py", "cloud = 17", 1) == renamed
    changed_other = workspace.save_named_console_script(other["id"], "other = 2", 0)
    assert changed_other["version"] == 1
    assert workspace.named_console_script(script_id) == renamed
    assert Workspace(tmp_path).named_console_script(script_id) == renamed


@pytest.mark.parametrize("change", ["stale", "collision", "missing_nonzero", "main_name"])
def test_desktop_cache_rejects_stale_or_colliding_import_without_overwriting(tmp_path, change):
    workspace = Workspace(tmp_path)
    first = workspace.create_console_script("first.py", "first = 1")
    second = workspace.create_console_script("second.py", "second = 2")
    first = workspace.save_named_console_script(first["id"], "first = 3", 0)
    before = workspace.console_scripts()
    code = "overwrite = 9"
    if change == "stale":
        args = (first["id"], first["name"], code, 0)
        expected_error = "VERSION_CONFLICT"
    elif change == "collision":
        args = (first["id"], second["name"], code, first["version"])
        expected_error = "SCRIPT_NAME_CONFLICT"
    elif change == "missing_nonzero":
        args = (uuid4().hex, "missing.py", code, 4)
        expected_error = "SCRIPT_ID_CONFLICT"
    else:
        args = ("analysis", "renamed.py", code, 0)
        expected_error = "SCRIPT_NAME_CONFLICT"
    with pytest.raises(DataError) as caught:
        workspace.cache_console_script(*args)
    assert caught.value.code == expected_error
    assert workspace.console_scripts() == before
    assert workspace.named_console_script(first["id"]) == first
    assert workspace.named_console_script(second["id"]) == second
    assert workspace.named_console_script("analysis")["code"] is None
