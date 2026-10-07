"""Project-local file layout metadata is bounded, private and never runs code."""
import json

from fastapi.testclient import TestClient

from openecon.desktop_runtime import create_desktop_app

A = "a" * 32
B = "b" * 32
SCRIPT = "c" * 32
FOLDER = "d" * 32


def open_project(client, project=A):
    route = f"/api/desktop/projects/{project}/workspace"
    response = client.get(route + "/session")
    assert response.status_code == 200
    return route, {"X-OpenEcon-Token": response.json()["token"]}


def cache(entries=None, **changes):
    return {"version": 0, "cloud_version": None, "base_entries": entries or [],
            "conflict": False, **changes}


def test_desktop_layout_uses_catalog_without_loading_or_executing_remote_code(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        path, headers = open_project(client)
        state = {"cloud_version": 1, "base_code": "", "role": "editor", "script_catalog": [
            {"id": "analysis", "name": "analysis.py", "version": 0},
            {"id": SCRIPT, "name": "remote.py", "version": 6},
        ]}
        assert client.put(path + "/desktop-sync-state", headers=headers, json=state).status_code == 200
        layout = client.get(path + "/desktop/file-layout", headers=headers).json()
        assert {item['id'] for item in layout['entries']} == {'analysis', SCRIPT}
        layout['entries'][1]['name'] = 'renamed.py'
        layout['entries'] = [{"kind": "folder", "id": FOLDER, "name": "Kod", "parent": None}, *layout['entries']]
        layout['entries'][2]['parent'] = FOLDER
        saved = client.put(path + "/desktop/file-layout", headers=headers, json=layout)
        assert saved.status_code == 200, saved.text
        assert saved.json()['version'] == 1
        assert client.get(path + f'/console/scripts/{SCRIPT}', headers=headers).status_code == 404
        assert app.state.desktop_projects.active_app.state.console._process is None
    with TestClient(create_desktop_app(tmp_path)) as client:
        path, headers = open_project(client)
        actual = client.get(path + '/desktop/file-layout', headers=headers).json()
        assert actual == saved.json()


def test_desktop_layout_denies_viewer_and_revoked_access(tmp_path):
    with TestClient(create_desktop_app(tmp_path)) as client:
        path, headers = open_project(client)
        layout = client.get(path + '/desktop/file-layout', headers=headers).json()
        for role, denied in [('viewer', False), ('editor', True)]:
            state = {'cloud_version': 0, 'base_code': '', 'role': role, 'access_denied': denied}
            assert client.put(path + '/desktop-sync-state', headers=headers, json=state).status_code == 200
            assert client.put(path + '/desktop/file-layout', headers=headers, json=layout).status_code == 403
        assert client.get(path + '/desktop/file-layout').status_code == 401


def test_sync_cache_compare_and_swap_persists_in_its_project(tmp_path):
    with TestClient(create_desktop_app(tmp_path)) as client:
        path, headers = open_project(client)
        route = path + '/desktop/file-layout-sync'
        assert client.get(route, headers=headers).json() is None
        first = client.put(route, headers=headers, json=cache(cloud_version=3)).json()
        assert first['version'] == 1
        assert client.put(route, headers=headers, json=cache()).status_code == 409
        assert client.get(route).status_code == 401
        other, other_headers = open_project(client, B)
        assert client.get(other + '/desktop/file-layout-sync', headers=other_headers).json() is None
    with TestClient(create_desktop_app(tmp_path)) as client:
        path, headers = open_project(client)
        assert client.get(path + '/desktop/file-layout-sync', headers=headers).json() == first


def test_sync_cache_rejects_paths_cycles_extra_fields_and_boolean_versions(tmp_path):
    with TestClient(create_desktop_app(tmp_path)) as client:
        path, headers = open_project(client)
        route = path + '/desktop/file-layout-sync'
        invalid = [cache(version=True), cache(cloud_version=-1), {**cache(), 'token': 'unaccepted'},
                   cache([{'kind': 'folder', 'id': FOLDER, 'name': '../unsafe', 'parent': None}]),
                   cache([{'kind': 'folder', 'id': FOLDER, 'name': 'self', 'parent': FOLDER}])]
        for body in invalid:
            assert client.put(route, headers=headers, json=body).status_code == 422
        assert client.get(route, headers=headers).json() is None


def test_sync_cache_does_not_follow_a_link(tmp_path):
    target = tmp_path / 'outside.json'
    target.write_text(json.dumps(cache()))
    with TestClient(create_desktop_app(tmp_path)) as client:
        path, headers = open_project(client)
        (tmp_path/'projects'/A/'desktop-file-layout-sync.json').symlink_to(target)
        assert client.get(path + '/desktop/file-layout-sync', headers=headers).status_code == 422
        assert json.loads(target.read_text()) == cache()
