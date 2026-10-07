"""Static transfer optimizations must preserve authenticated response boundaries."""
from importlib.resources import files as resource_files

from fastapi.testclient import TestClient
import pytest

from openecon.team_auth import TeamAuth
from openecon.team_server import create_team_app


@pytest.fixture
def static_client(tmp_path, monkeypatch):
    static = tmp_path / 'static'
    assets = static / 'assets'
    assets.mkdir(parents=True)
    (static / 'index.html').write_text('<html>OpenEcon</html>')
    source = 'const data = "navy";\n' * 1000
    (assets / 'index-1234abcd.js').write_text(source)
    (assets / 'unversioned.js').write_text(source)
    monkeypatch.setattr('openecon.team_server.files',
                        lambda package: tmp_path if package == 'openecon' else resource_files(package))
    origin = 'https://app.example.com'
    app = create_team_app(store=None, storage=None, runner=None,
        auth=TeamAuth('openecon-test'), public_origin=origin,
        firebase_config={'projectId': 'openecon-test', 'authDomain': 'openecon-test.firebaseapp.com'})
    with TestClient(app, base_url=origin) as client:
        yield client, source


def test_content_hashed_assets_are_compressed_and_cached_with_security_headers(static_client):
    client, source = static_client
    response = client.get('/assets/index-1234abcd.js', headers={'Accept-Encoding': 'gzip'})
    assert response.status_code == 200
    assert response.text == source
    assert response.headers['content-encoding'] == 'gzip'
    assert 'accept-encoding' in response.headers['vary'].lower()
    assert response.headers['cache-control'] == 'public, max-age=31536000, immutable'
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert "frame-ancestors 'none'" in response.headers['content-security-policy']
    conditional = client.get('/assets/index-1234abcd.js',
                             headers={'If-None-Match': response.headers['etag']})
    assert conditional.status_code == 304
    assert conditional.headers['cache-control'] == response.headers['cache-control']


def test_identity_transfer_and_unversioned_files_do_not_receive_immutable_cache(static_client):
    client, source = static_client
    response = client.get('/assets/index-1234abcd.js', headers={'Accept-Encoding': 'identity'})
    assert response.text == source
    assert 'content-encoding' not in response.headers
    for path in ('/', '/assets/unversioned.js', '/assets/missing-1234abcd.js'):
        assert 'immutable' not in client.get(path).headers.get('cache-control', '')
    assert client.get('/').headers['cache-control'] == 'no-cache'


def test_auth_configuration_and_denied_requests_remain_uncached(static_client):
    client, _ = static_client
    config = client.get('/api/auth/config')
    assert config.status_code == 200
    assert config.headers['cache-control'] == 'no-store'
    denied = client.get('/api/projects')
    assert denied.status_code == 401
    assert denied.headers['cache-control'] == 'no-store'
