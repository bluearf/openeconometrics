"""Accepted work remains stoppable during sandbox cutover and jobs rollback."""
from unittest.mock import Mock

import pytest

from openecon.team_dispatch import MigratingSandboxRunner
from openecon.team_runner import GoogleJobRunner


def test_new_launch_uses_sandbox_while_existing_jobs_keep_status_and_cancel():
    sandbox, legacy = Mock(), Mock()
    runner = MigratingSandboxRunner(sandbox, legacy)
    runner.start('private-input', timeout_seconds=60)
    sandbox.start.assert_called_once_with('private-input', timeout_seconds=60)
    legacy.start.assert_not_called()
    old_operation = 'projects/openecon-test/locations/us-central1/operations/old'
    old_execution = 'projects/openecon-test/locations/us-central1/jobs/compute/executions/old'
    runner.status(old_operation)
    runner.cancel(old_execution)
    legacy.status.assert_called_once_with(old_operation)
    legacy.cancel.assert_called_once_with(old_execution)
    sandbox.status.assert_not_called()
    sandbox.cancel.assert_not_called()


def test_sandbox_handles_keep_the_new_parent_completion_boundary():
    sandbox, legacy = Mock(), Mock()
    runner = MigratingSandboxRunner(sandbox, legacy)
    handle = 'sandbox:' + 'a' * 32 + ':' + 'b' * 32
    runner.status(handle)
    runner.cancel(handle)
    sandbox.status.assert_called_once_with(handle)
    sandbox.cancel.assert_called_once_with(handle)
    legacy.status.assert_not_called()
    legacy.cancel.assert_not_called()


def test_jobs_rollback_changes_new_launch_only_and_keeps_sandbox_cleanup_boundary():
    sandbox, legacy = Mock(), Mock()
    sandbox.status.return_value = {'status': 'running', 'cleanup_confirmed': False}
    runner = MigratingSandboxRunner(sandbox, legacy, start_backend='jobs')
    assert runner.start('private-input', timeout_seconds=37) is legacy.start.return_value
    legacy.start.assert_called_once_with('private-input', timeout_seconds=37)
    sandbox.start.assert_not_called()
    handle = 'sandbox:' + 'a' * 32 + ':' + 'b' * 32
    assert runner.status(handle) == {'status': 'running', 'cleanup_confirmed': False}
    runner.cancel(handle)
    sandbox.status.assert_called_once_with(handle)
    sandbox.cancel.assert_called_once_with(handle)
    legacy.status.assert_not_called()
    legacy.cancel.assert_not_called()
    old_operation = 'projects/openecon-test/locations/us-central1/operations/old'
    old_execution = 'projects/openecon-test/locations/us-central1/jobs/compute/executions/old'
    runner.status(old_operation)
    runner.cancel(old_execution)
    legacy.status.assert_called_once_with(old_operation)
    legacy.cancel.assert_called_once_with(old_execution)


def test_launch_error_never_falls_back_or_retries_on_other_backend():
    sandbox, legacy = Mock(), Mock()
    legacy.start.side_effect = RuntimeError('Launch outcome unknown')
    runner = MigratingSandboxRunner(sandbox, legacy, start_backend='jobs')
    with pytest.raises(RuntimeError, match='unknown'):
        runner.start('private-input')
    assert legacy.start.call_count == 1
    sandbox.start.assert_not_called()


@pytest.mark.parametrize('backend,legacy', [('jobs', None), ('local', Mock()), (None, Mock())])
def test_invalid_or_unconfigured_start_backend_fails_closed(backend, legacy):
    with pytest.raises(ValueError):
        MigratingSandboxRunner(Mock(), legacy, start_backend=backend)


def test_foreign_or_unknown_job_handles_still_fail_legacy_scope_checks(monkeypatch):
    from openecon import team_runner
    jobs, executions = Mock(), Mock()
    monkeypatch.setattr(team_runner, '_clients', lambda: (jobs, executions))
    legacy = GoogleJobRunner('openecon-test', 'us-central1', 'compute')
    runner = MigratingSandboxRunner(Mock(), legacy, start_backend='jobs')
    for handle in ('unknown', 'projects/foreign-project/locations/us-central1/operations/stolen'):
        with pytest.raises(ValueError):
            runner.status(handle)
    with pytest.raises(ValueError):
        runner.cancel('projects/openecon-test/locations/us-central1/jobs/other/executions/stolen')
    jobs.transport.operations_client.get_operation.assert_not_called()
    executions.cancel_execution.assert_not_called()
