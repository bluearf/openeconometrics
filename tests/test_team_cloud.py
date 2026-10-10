"""Production wiring builds a sync-only backend and never a compute runner."""
import json
import os
from unittest.mock import Mock

import pytest

from openecon.cloud import cloud_app
from openecon.team_cloud import REQUIRED_SETTINGS, team_app

COMMIT = '0123456789abcdef0123456789abcdef01234567'


@pytest.fixture
def configuration(monkeypatch):
    from openecon import team_auth, team_server, team_storage, team_store
    for key in tuple(os.environ):
        if key.startswith('OPENECON_'):
            monkeypatch.delenv(key)
    settings = {
        'OPENECON_PROJECT_ID': 'openecon-test',
        'OPENECON_PUBLIC_ORIGIN': 'https://openecon.example',
        'OPENECON_OWNER_EMAIL': 'owner@example.com',
        'OPENECON_BUCKET': 'openecon-projects',
        'OPENECON_SIGNER_EMAIL': 'signer@openecon-test.iam.gserviceaccount.com',
        'OPENECON_FIREBASE_CONFIG': json.dumps({'projectId': 'openecon-test', 'apiKey': 'key',
                                               'authDomain': 'openecon-test.firebaseapp.com',
                                               'appId': 'application'}),
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    created = Mock(name='create_team_app', side_effect=lambda **kwargs: kwargs)
    monkeypatch.setattr(team_auth, 'TeamAuth', Mock())
    monkeypatch.setattr(team_storage, 'TeamStorage', Mock())
    monkeypatch.setattr(team_store, 'FirestoreDocuments', Mock())
    monkeypatch.setattr(team_store, 'TeamStore', Mock())
    monkeypatch.setattr(team_server, 'create_team_app', created)
    return created


def test_sync_backend_needs_only_identity_and_storage_settings(configuration):
    kwargs = team_app()
    assert set(kwargs) == {'store', 'storage', 'auth', 'public_origin', 'firebase_config',
                           'source_commit_id'}
    assert 'runner' not in kwargs
    assert kwargs['public_origin'] == 'https://openecon.example'
    assert kwargs['source_commit_id'] is None


def test_build_source_commit_reaches_the_app(monkeypatch, configuration):
    monkeypatch.setenv('OPENECON_SOURCE_COMMIT', COMMIT)
    assert team_app()['source_commit_id'] == COMMIT


@pytest.mark.parametrize('legacy', [
    {'OPENECON_RUNNER': 'sandbox'}, {'OPENECON_RUNNER': 'jobs'},
    {'OPENECON_COMPUTE_JOB': 'openecon-compute',
     'OPENECON_COMPUTE_EMAIL': 'compute@openecon-test.iam.gserviceaccount.com',
     'OPENECON_COMPUTE_SERVICE': 'openecon-sandbox',
     'OPENECON_COMPUTE_ORIGIN': 'https://openecon-sandbox-123456789012.us-central1.run.app',
     'OPENECON_COMPUTE_IMAGE': 'image@sha256:' + 'a' * 64,
     'OPENECON_BROKER_CALLER_SUB': '108062605432490492564'},
])
def test_retired_compute_settings_never_create_an_execution_backend(monkeypatch, configuration, legacy):
    for key, value in legacy.items():
        monkeypatch.setenv(key, value)
    kwargs = team_app()
    assert 'runner' not in kwargs
    assert not any('compute' in key or 'runner' in key for key in kwargs)


@pytest.mark.parametrize('missing', REQUIRED_SETTINGS)
def test_missing_sync_configuration_fails_closed(monkeypatch, configuration, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match='identity and storage'):
        team_app()
    configuration.assert_not_called()


def test_firebase_configuration_must_belong_to_the_project(monkeypatch, configuration):
    monkeypatch.setenv('OPENECON_FIREBASE_CONFIG', json.dumps({
        'projectId': 'another-project', 'apiKey': 'key',
        'authDomain': 'another-project.firebaseapp.com', 'appId': 'application'}))
    with pytest.raises(ValueError, match='Firebase'):
        team_app()
    configuration.assert_not_called()


@pytest.mark.parametrize('mode', [None, '', 'single-owner', 'sandbox-broker', 'local'])
def test_cloud_entry_point_refuses_the_retired_cloud_workbench(monkeypatch, configuration, mode):
    if mode is None:
        monkeypatch.delenv('OPENECON_MODE', raising=False)
    else:
        monkeypatch.setenv('OPENECON_MODE', mode)
    with pytest.raises(ValueError, match='desktop app'):
        cloud_app()
    configuration.assert_not_called()


def test_cloud_entry_point_serves_team_sync(monkeypatch, configuration):
    monkeypatch.setenv('OPENECON_MODE', 'teams')
    assert 'runner' not in cloud_app()
    configuration.assert_called_once()
