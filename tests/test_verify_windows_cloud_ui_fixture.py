"""Run the controller's exact Python fixture through the real local worker.

This checks the fixture/archive protocol before hosted native QA, not Windows
installation, provider login, connectivity or member readback acceptance.
"""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from openecon.console import ConsoleSession
from openecon.team_server import validate_worker_result
from openecon.workspace import Workspace


def test_hosted_controller_fixture_is_one_real_hc3_run_and_preserves_archive_payload(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required to import the exact native controller fixture.')
    controller = Path(__file__).parents[1] / 'desktop/scripts/verify_windows_cloud_ui.mjs'
    script = f'import {{EXECUTION_COMMAND}} from {json.dumps(controller.as_uri())}; process.stdout.write(EXECUTION_COMMAND);'
    code = subprocess.check_output([node, '--input-type=module', '-e', script], text=True)
    workspace = Workspace(tmp_path / 'qa-native-fixture')
    session = ConsoleSession(workspace)
    try:
        result = session.execute(code)
        assert result['status'] == 'ok', result['error']
        assert result['stdout'].splitlines() == [
            'MARKET87_WINDOWS_NATIVE_SHARE_OK', 'MARKET87_NATIVE_EXECUTIONS=1']
        assert [output['type'] for output in result['outputs']] == ['model', 'latex']
        model, publication = result['outputs']
        assert model['data']['nobs'] == 480 and model['data']['spec']['covariance'] == 'HC3'
        assert publication['data'] == model['latex']
        assert result['events'] == [
            {'type': 'output', 'index': 0}, {'type': 'output', 'index': 1},
            {'type': 'stdout', 'text': result['stdout']},
        ]
        protocol_script = (
            f'import {{fixtureRecordMatches}} from {json.dumps(controller.as_uri())}; '
            'let input=""; for await (const chunk of process.stdin) input+=chunk; '
            'process.stdout.write(JSON.stringify(fixtureRecordMatches(JSON.parse(input))));'
        )
        assert subprocess.check_output(
            [node, '--input-type=module', '-e', protocol_script],
            input=json.dumps(result), text=True,
        ) == 'true'
        run = {'id': '1' * 32, 'code': code, 'created_at': result['created_at'],
               'generation': 0, 'email': 'qa-viewer@example.com'}
        # A cloud request validates its own JSON copy, never the local record.
        archive, generated = validate_worker_result(
            {'execution_id': run['id'], 'record': json.loads(json.dumps(result))}, run)
        assert generated == []
        assert subprocess.check_output(
            [node, '--input-type=module', '-e', protocol_script],
            input=json.dumps(archive), text=True,
        ) == 'true'
        for key in ['code', 'status', 'stdout', 'events']:
            assert archive[key] == result[key]
        original_outputs = json.loads(json.dumps(result['outputs']))
        assert archive['outputs'][1] == original_outputs[1]
        original_model, archived_model = original_outputs[0], archive['outputs'][0]
        assert set(archived_model['data']['display_omitted']) == set(original_model['data']['display_omitted'])
        original_model['data']['display_omitted'] = sorted(original_model['data']['display_omitted'])
        archived_model['data']['display_omitted'] = sorted(archived_model['data']['display_omitted'])
        assert archived_model == original_model
        assert (workspace.path / 'market87-native-execution-count.txt').read_text() == '1'
    finally:
        session.close()
    reopened = ConsoleSession(Workspace(workspace.path))
    try:
        history = reopened.snapshot()['history']
        assert len(history) == 1 and history[0]['id'] == result['id']
        assert history[0]['outputs'] == result['outputs']
        assert history[0]['events'] == result['events']
        assert reopened.status()['pid'] is None
        assert (workspace.path / 'market87-native-execution-count.txt').read_text() == '1'
    finally:
        reopened.close()
