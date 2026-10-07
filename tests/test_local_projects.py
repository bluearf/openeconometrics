"""Fresh accountless workspaces and their separation from cached team access."""
import json
from fastapi.testclient import TestClient
from openecon.desktop_runtime import create_desktop_app


def catalog_headers(client):
    return {"X-OpenEcon-Token": client.get('/api/desktop/session').json()['token']}


def test_fresh_local_creation_torch_and_restart_preserve_code_data_environment_history(tmp_path):
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = catalog_headers(client)
        assert client.get('/api/desktop/local-projects').status_code == 401
        assert client.post('/api/desktop/local-projects', headers=headers, json={'name': ' '}).status_code == 422
        created = client.post('/api/desktop/local-projects', headers=headers,
                              json={'name': 'Offline research', 'description': 'Synthetic only'}).json()
        ident = created['id']
        assert client.post(f'/api/desktop/local-projects/{ident}/open', headers=headers).json() == created
        prefix = f'/api/desktop/projects/{ident}/workspace'
        project_headers = {'X-OpenEcon-Token': client.get(prefix + '/session').json()['token']}
        code = 'import openecon as oe\nmodel = oe.ols(data=oe.example(), y="wage", x=["education","experience"], covariance="HC3")\nprint(model.nobs)\ndisplay(model)'
        saved = client.put(prefix + '/console/scripts/analysis', headers=project_headers, json={'code':code,'version':0}).json()
        uploaded = client.post(prefix + '/datasets/upload', headers=project_headers,
                               files={'file':('synthetic.csv',b'x,y\n1,2\n2,3\n3,4\n','text/csv')}).json()
        environment = client.get(prefix + '/environment', headers=project_headers).json()
        run = client.post(prefix + '/console/execute', headers=project_headers, json={'code':code}).json()
        assert run['status'] == 'ok',run
        assert '480' in run['stdout']
        assert client.get(prefix + '/desktop-outbox', headers=project_headers).json() == {'items':[]}
        assert not (tmp_path / 'projects' / ident / 'desktop-sync-state.json').exists()
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = catalog_headers(client)
        assert client.get('/api/desktop/local-projects',headers=headers).json() == {'projects':[created]}
        assert client.post(f'/api/desktop/local-projects/{ident}/open',headers=headers).status_code == 200
        project_headers = {'X-OpenEcon-Token':client.get(prefix + '/session').json()['token']}
        assert client.get(prefix + '/console/scripts/analysis',headers=project_headers).json() == saved
        assert client.get(prefix + '/datasets',headers=project_headers).json()['datasets'][0]['data_hash'] == uploaded['data_hash']
        assert client.get(prefix + '/environment',headers=project_headers).json()['manifest'] == environment['manifest']
        history = client.get(prefix + '/console',headers=project_headers).json()['history']
        assert history[0]['id'] == run['id'] and history[0]['outputs'] == run['outputs']


def test_cloud_or_revoked_project_cannot_be_registered_or_opened_as_local(tmp_path):
    cloud = 'a' * 32
    app = create_desktop_app(tmp_path)
    path = app.state.desktop_projects.project_path(cloud)
    (path / 'desktop-sync-state.json').write_text(json.dumps({'cloud_version':1,'base_code':'private','role':'editor','access_denied':True}))
    with TestClient(app) as client:
        headers = catalog_headers(client)
        assert client.post('/api/desktop/local-projects',headers=headers,json={'name':'Attack','id':cloud}).status_code == 422
        assert client.post(f'/api/desktop/local-projects/{cloud}/open',headers=headers).status_code == 404
        (tmp_path / 'local-projects.json').write_text(json.dumps({'projects':[{'id':cloud,'name':'Tampered','description':''}]}))
        assert client.post(f'/api/desktop/local-projects/{cloud}/open',headers=headers).status_code == 403
        assert 'private' in (path / 'desktop-sync-state.json').read_text()


def test_local_catalog_symlinks_corruption_and_cross_site_access_are_refused(tmp_path):
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = catalog_headers(client)
        assert client.get('/api/desktop/local-projects',headers={**headers,'Origin':'https://attacker.invalid'}).status_code == 403
        path=tmp_path/'local-projects.json'
        path.write_text('{invalid')
        assert client.get('/api/desktop/local-projects',headers=headers).status_code == 422
        path.unlink()
        other=tmp_path/'other.json'
        other.write_text('{"projects":[]}')
        path.symlink_to(other)
        assert client.get('/api/desktop/local-projects',headers=headers).status_code == 422
