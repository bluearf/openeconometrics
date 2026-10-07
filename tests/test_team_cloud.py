"""Deployment selection affects new launches, never the owner of saved handles."""
import json
import os
from unittest.mock import Mock

import pytest

from openecon.team_cloud import team_app
from openecon.team_dispatch import MigratingSandboxRunner
from openecon.team_sandbox_runner import GoogleSandboxRunner


@pytest.fixture
def configuration(monkeypatch):
    from openecon import team_auth, team_runner, team_sandbox_runner, team_server, team_storage, team_store
    for key in tuple(os.environ):
        if key.startswith('OPENECON_'):
            monkeypatch.delenv(key)
    settings = {
        'OPENECON_PROJECT_ID': 'openecon-test',
        'OPENECON_PUBLIC_ORIGIN': 'https://openecon.example',
        'OPENECON_OWNER_EMAIL': 'owner@example.com',
        'OPENECON_BUCKET': 'openecon-runs',
        'OPENECON_SIGNER_EMAIL': 'signer@openecon-test.iam.gserviceaccount.com',
        'OPENECON_COMPUTE_EMAIL': 'compute@openecon-test.iam.gserviceaccount.com',
        'OPENECON_COMPUTE_JOB': 'openecon-compute',
        'OPENECON_FIREBASE_CONFIG': json.dumps({'projectId': 'openecon-test', 'apiKey': 'key',
                                               'authDomain': 'openecon-test.firebaseapp.com',
                                               'appId': 'application'}),
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    sandbox, jobs = Mock(name='sandbox'), Mock(name='jobs')
    sandbox_factory, jobs_factory = Mock(return_value=sandbox), Mock(return_value=jobs)
    monkeypatch.setattr(team_sandbox_runner, 'GoogleSandboxRunner', sandbox_factory)
    monkeypatch.setattr(team_runner, 'GoogleJobRunner', jobs_factory)
    monkeypatch.setattr(team_auth, 'TeamAuth', Mock())
    monkeypatch.setattr(team_storage, 'TeamStorage', Mock())
    monkeypatch.setattr(team_store, 'FirestoreDocuments', Mock())
    monkeypatch.setattr(team_store, 'TeamStore', Mock())
    monkeypatch.setattr(team_server, 'create_team_app', lambda **kwargs: kwargs['runner'])
    return sandbox, jobs, sandbox_factory, jobs_factory


def add_sandbox(monkeypatch):
    for key, value in {
        'OPENECON_COMPUTE_SERVICE': 'openecon-sandbox',
        'OPENECON_COMPUTE_ORIGIN': 'https://openecon-sandbox-123456789012.us-central1.run.app',
        'OPENECON_COMPUTE_IMAGE': 'image@sha256:' + 'a' * 64,
        'OPENECON_BROKER_CALLER_SUB': '108062605432490492564',
    }.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize('explicit_jobs', [False, True])
def test_existing_jobs_deployments_without_sandbox_configuration_unchanged(monkeypatch, configuration, explicit_jobs):
    sandbox, jobs, sandbox_factory, jobs_factory = configuration
    if explicit_jobs:
        monkeypatch.setenv('OPENECON_RUNNER', 'jobs')
    assert team_app() is jobs
    sandbox_factory.assert_not_called()
    jobs_factory.assert_called_once_with('openecon-test', 'us-central1', 'openecon-compute',
        service_account='compute@openecon-test.iam.gserviceaccount.com')


@pytest.mark.parametrize('backend', ['jobs', 'sandbox'])
def test_cutover_and_rollback_keep_both_execution_handle_readers(monkeypatch, configuration, backend):
    sandbox, jobs, sandbox_factory, jobs_factory = configuration
    add_sandbox(monkeypatch)
    monkeypatch.setenv('OPENECON_RUNNER', backend)
    runner = team_app()
    assert isinstance(runner, MigratingSandboxRunner)
    runner.start('input', timeout_seconds=15)
    (jobs if backend == 'jobs' else sandbox).start.assert_called_once_with('input', timeout_seconds=15)
    (sandbox if backend == 'jobs' else jobs).start.assert_not_called()
    handle = 'sandbox:' + 'a' * 32 + ':' + 'b' * 32
    runner.status(handle)
    runner.cancel(handle)
    sandbox.status.assert_called_once_with(handle)
    sandbox.cancel.assert_called_once_with(handle)
    jobs.status.assert_not_called()
    jobs.cancel.assert_not_called()
    assert jobs_factory.call_count == sandbox_factory.call_count == 1
    assert sandbox_factory.call_args.kwargs['caller_subject'] == '108062605432490492564'


def test_sandbox_only_deployment_does_not_require_legacy_job(monkeypatch, configuration):
    _, _, _, jobs_factory = configuration
    add_sandbox(monkeypatch)
    monkeypatch.setenv('OPENECON_RUNNER', 'sandbox')
    monkeypatch.delenv('OPENECON_COMPUTE_JOB')
    runner = team_app()
    with pytest.raises(ValueError, match='legacy'):
        runner.status('projects/openecon-test/locations/us-central1/operations/old')
    jobs_factory.assert_not_called()


def test_rollback_requires_configured_legacy_job(monkeypatch, configuration):
    add_sandbox(monkeypatch)
    monkeypatch.setenv('OPENECON_RUNNER', 'jobs')
    monkeypatch.delenv('OPENECON_COMPUTE_JOB')
    with pytest.raises(ValueError, match='launch backend'):
        team_app()


@pytest.mark.parametrize('missing', ['OPENECON_COMPUTE_SERVICE', 'OPENECON_COMPUTE_ORIGIN',
                                    'OPENECON_COMPUTE_IMAGE'])
def test_partial_sandbox_configuration_cannot_silently_discard_accepted_handles(monkeypatch, configuration, missing):
    add_sandbox(monkeypatch)
    monkeypatch.setenv('OPENECON_RUNNER', 'jobs')
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match='Sandbox compute requires'):
        team_app()
    configuration[2].assert_not_called()
    configuration[3].assert_not_called()


def test_unsupported_backend_fails_even_with_complete_sandbox_settings(monkeypatch, configuration):
    add_sandbox(monkeypatch)
    monkeypatch.setenv('OPENECON_RUNNER', 'local')
    with pytest.raises(ValueError, match='supported isolated'):
        team_app()
    configuration[2].assert_not_called()
    configuration[3].assert_not_called()


@pytest.mark.parametrize('backend', ['sandbox', 'jobs'])
def test_real_runner_rejects_missing_caller_subject_for_cutover_and_rollback(monkeypatch, configuration, backend):
    from openecon import team_sandbox_runner
    add_sandbox(monkeypatch)
    monkeypatch.setenv('OPENECON_RUNNER', backend)
    monkeypatch.delenv('OPENECON_BROKER_CALLER_SUB')
    monkeypatch.setattr(team_sandbox_runner, 'GoogleSandboxRunner', GoogleSandboxRunner)
    with pytest.raises(ValueError, match='numeric identity'):
        team_app()
    configuration[3].assert_not_called()
