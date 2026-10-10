"""Reject stale workflow identities before numerical environments are installed."""
import subprocess
from pathlib import Path

import pytest

from scripts import verify_ci_preflight as preflight


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, 'init', '--quiet')
    git(tmp_path, 'config', 'user.email', 'ci-fixture@example.invalid')
    git(tmp_path, 'config', 'user.name', 'CI fixture')
    (tmp_path / 'source.txt').write_text('reviewed\n')
    git(tmp_path, 'add', 'source.txt')
    git(tmp_path, 'commit', '--quiet', '-m', 'reviewed source')
    return tmp_path


def environment(root, event='workflow_dispatch'):
    return {'GITHUB_SHA': git(root, 'rev-parse', 'HEAD'), 'GITHUB_EVENT_NAME': event,
            'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1'}


@pytest.mark.parametrize('event', ['schedule', 'workflow_dispatch'])
def test_daily_preflight_accepts_actual_clean_source(repository, event):
    report = preflight.verify(repository, environment(repository, event))
    assert report['status'] == 'passed'
    assert report['verified_git']['source_tree'] == git(repository, 'rev-parse', 'HEAD^{tree}')


def test_preflight_rejects_stale_source_before_installation(repository):
    values = environment(repository)
    (repository / 'source.txt').write_text('new source\n')
    git(repository, 'commit', '--quiet', '-am', 'new source')
    with pytest.raises(ValueError, match='checkout differs'):
        preflight.verify(repository, values)


def test_preflight_rejects_tracked_edits(repository):
    values = environment(repository)
    (repository / 'source.txt').write_text('unreviewed\n')
    with pytest.raises(ValueError, match='tracked source differs'):
        preflight.verify(repository, values)


def test_daily_workflow_stops_sdk_provisioning_when_preflight_fails():
    workflow = (Path(__file__).resolve().parents[1] / '.github/workflows/ci.yml').read_text()
    sdk = workflow.split('\n  sdk-components:\n', 1)[1].split('\n  merge-gate:\n', 1)[0]
    assert '    needs: preflight\n' in sdk
    assert 'if: always()' not in sdk.split('    steps:', 1)[0]
    assert 'continue-on-error:' not in workflow.split('\n  sdk-components:\n', 1)[0]
    assert 'scripts/verify_ci_preflight.py --report' in workflow


@pytest.mark.parametrize('versions, passes', [(['2.14.0'], True),
                                               (['2.14.0+cpu', '2.14.0'], True),
                                               (['2.14.0+cpu', '2.15.0'], False)])
def test_container_cpu_selection_handles_lock_variants(tmp_path, versions, passes):
    import json
    root = Path(__file__).resolve().parents[1]
    block = (root / 'Dockerfile').read_text().split("RUN python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    (tmp_path / 'uv.lock').write_text(''.join(
        '[[package]]\nname = "torch"\nversion = ' + json.dumps(version) + '\n'
        for version in versions))
    (tmp_path / 'production-all.txt').write_text('torch==2.14.0\nnvidia-cublas==1\npandas==2.3.4\n')
    # Run the actual container preparation in an owned temporary filesystem.
    block = block.replace('/tmp/', str(tmp_path) + '/')
    result = subprocess.run(['python3', '-c', block], cwd=tmp_path, capture_output=True, text=True)
    assert (result.returncode == 0) == passes, result.stderr
    if passes:
        assert (tmp_path / 'torch-version.txt').read_text() == '2.14.0'
        assert (tmp_path / 'production-cpu.txt').read_text() == 'pandas==2.3.4\n'
