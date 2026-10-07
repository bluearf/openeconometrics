"""Platform waiting has its own bounded lease, separate from the Python limit."""
from datetime import datetime, timedelta, timezone

import pytest

from test_team_server import api as api, header


@pytest.mark.parametrize('platform_status', ['queued', 'running'])
def test_platform_wait_beyond_five_minutes_keeps_project_lock_and_never_launches_again(
        api, monkeypatch, platform_status):
    clock = [datetime(2026, 10, 1, 10, tzinfo=timezone.utc)]
    monkeypatch.setattr('openecon.team_store.now', lambda: clock[0].isoformat())
    monkeypatch.setattr('openecon.team_server.now', lambda: clock[0].isoformat())
    run = api.store.begin_run(api.pid, api.users['editor'], 'print(1)', 60)
    api.store.update_run(api.pid, run['id'], operation='operation-1',
                         execution='execution-1', state='running')
    api.runner.status_value = platform_status
    clock[0] += timedelta(seconds=301)
    response = api.client.get(api.prefix + '/console', headers=header('editor'))
    assert response.status_code == 200
    assert response.json()['status']['running'] is True
    assert response.json()['history'] == []
    retry = api.client.post(api.prefix + '/console/execute', headers=header('editor'),
                            json={'code': 'print(2)'})
    assert retry.status_code == 409
    assert api.runner.starts == api.runner.cancellations == []
    assert api.store.project(api.pid, api.users['editor'])['active_run'] == run['id']
    assert run['id'] in api.store.db.get('oe_limits/compute')['active']


def test_total_wait_expires_cancels_and_releases_both_leases_without_refunding_quota(api, monkeypatch):
    clock = [datetime(2026, 10, 1, 10, tzinfo=timezone.utc)]
    monkeypatch.setattr('openecon.team_store.now', lambda: clock[0].isoformat())
    monkeypatch.setattr('openecon.team_server.now', lambda: clock[0].isoformat())
    run = api.store.begin_run(api.pid, api.users['editor'], 'print(1)', 60)
    api.store.update_run(api.pid, run['id'], operation='operation-1',
                         execution='execution-1', state='running')
    api.runner.status_value = 'queued'
    clock[0] += timedelta(seconds=1081)
    response = api.client.get(api.prefix + '/console', headers=header('editor'))
    assert response.status_code == 200
    record = response.json()['history'][-1]
    assert record['status'] == 'timeout'
    assert 'total wait time' in record['error']['message']
    assert record['error']['message'] != 'The computation timed out.'
    assert api.runner.cancellations == ['execution-1']
    assert response.json()['status']['running'] is False
    assert api.store.db.get('oe_limits/compute')['active'] == {}
    assert api.store.db.get('oe_limits/compute')['runs'] == 1
    assert api.store.project(api.pid, api.users['editor'])['daily_runs'] == 1
    api.runner.status_value = 'succeeded'
    replacement = api.client.post(api.prefix + '/console/execute', headers=header('editor'),
                                  json={'code': 'print(2)', 'wait_for_result': False})
    assert replacement.status_code == 202
    history = api.client.get(api.prefix + '/console', headers=header('editor')).json()['history']
    assert history[-1]['status'] == 'ok'
    assert len(api.runner.starts) == 1


def test_python_timeout_keeps_its_own_limit_and_result_instead_of_becoming_queue_timeout(api):
    message = 'Python exceeded the time limit. The worker was stopped and session variables were cleared.'
    def timeout(payload):
        payload['record'].update(status='timeout', error={
            'type': 'TIMEOUT', 'message': message, 'traceback': ''})
    api.runner.output = timeout
    response = api.client.post(api.prefix + '/console/execute', headers=header('editor'),
                               json={'code': 'while True: pass', 'timeout_seconds': 60,
                                     'wait_for_result': False})
    assert response.status_code == 202
    record = api.client.get(api.prefix + '/console', headers=header('editor')).json()['history'][-1]
    assert record['status'] == 'timeout'
    assert record['error']['message'] == message
    assert api.runner.starts[0][1] == 60
    assert api.storage.current_run['timeout_seconds'] == 60
    assert api.runner.cancellations == []
    assert api.store.db.get('oe_limits/compute')['active'] == {}


def test_explicit_cancel_remains_interrupted_when_total_wait_deadline_has_passed(api, monkeypatch):
    run = api.store.begin_run(api.pid, api.users['editor'], 'print(1)', 60)
    api.store.update_run(api.pid, run['id'], operation='operation-1',
                         execution='execution-1', state='running')
    api.runner.status_value = 'queued'
    expired = datetime.fromisoformat(run['deadline']) + timedelta(seconds=1)
    monkeypatch.setattr('openecon.team_server.now', lambda: expired.isoformat())
    response = api.client.post(api.prefix + '/console/interrupt', headers=header('editor'))
    assert response.status_code == 200 and response.json()['status'] == 'interrupted'
    stored = api.store.run(api.pid, run['id'])
    assert stored['state'] == 'cancelled'
    assert stored['record_summary']['status'] == 'interrupted'
    assert api.runner.cancellations == ['execution-1']
    assert api.store.db.get('oe_limits/compute')['active'] == {}
