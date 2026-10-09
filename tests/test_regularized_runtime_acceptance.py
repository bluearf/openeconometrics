"""Actual source console proof plumbing; these tests do not claim frozen Windows QA."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from openecon.console import ConsoleSession
from openecon.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'regularized_runtime_qa', ROOT / 'scripts/verify_regularized_all_family_runtime.py')
QA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QA)


@pytest.fixture(scope='module')
def actual_source_results(tmp_path_factory):
    root = tmp_path_factory.mktemp('regularized-console-proof')
    workspace = Workspace(root)
    session = ConsoleSession(workspace)
    setup = QA.SETUP.replace("assert getattr(sys,'frozen',False)\n", '')
    code = setup + (ROOT / 'docs/examples/regularized_all_family.py').read_text() + QA.SAVE
    try:
        run = session.execute(code, timeout_seconds=120)
        proof = QA.verify_run(run, root, require_frozen=False)
        saved = workspace.console_history()[0]
    finally:
        session.close()
    cold = ConsoleSession(Workspace(root))
    try:
        assert cold.status()['pid'] is None
        reopened = cold.snapshot()['history'][0]
        assert {key: reopened[key] for key in ('code', 'stdout', 'outputs', 'events')} == {
            key: saved[key] for key in ('code', 'stdout', 'outputs', 'events')}
        replay_code = QA.REPLAY.replace("assert getattr(sys,'frozen',False)\n", '')
        replay = cold.execute(replay_code, timeout_seconds=120)
        restored = QA.verify_run(replay, root, require_frozen=False, order=sorted(QA.NAMES))
        assert restored == proof
    finally:
        cold.close()
    return root, run, replay, proof


def test_complete_state_above_stdout_bound_uses_small_exact_hash_marker(actual_source_results):
    root, run, replay, proof = actual_source_results
    assert (root / QA.ARTIFACTS[0]).stat().st_size > 64 * 1024
    assert len(run['stdout'].encode()) < 4096
    assert len(replay['stdout'].encode()) < 4096
    assert set(proof['models']) == set(QA.NAMES)
    assert len(proof['artifact_manifest']) == 10
    # A source worker can never satisfy the frozen acceptance gate.
    with pytest.raises(AssertionError):
        QA.verify_run(run, root)


@pytest.mark.parametrize('damage', ['drop_row', 'wrong_estimate', 'no_latex', 'duplicate_marker'])
def test_real_serialized_table_damage_is_rejected(actual_source_results, damage):
    root, original, _, _ = actual_source_results
    run = deepcopy(original)
    if damage == 'drop_row':
        run['outputs'][0]['data']['rows'].pop()
    elif damage == 'wrong_estimate':
        run['outputs'][0]['data']['rows'][0][1] += 1
    elif damage == 'no_latex':
        run['outputs'][0].pop('latex')
    else:
        marker = next(line for line in run['stdout'].splitlines() if line.startswith(QA.MARKER))
        run['stdout'] += marker + '\n'
    with pytest.raises(AssertionError):
        QA.verify_run(run, root, require_frozen=False)


def test_saved_artifact_mutation_is_rejected(actual_source_results):
    root, run, _, _ = actual_source_results
    path = root / 'regularized-all-family-ridge.tex'
    before = path.read_bytes()
    try:
        path.write_bytes(before + b'changed')
        with pytest.raises(AssertionError):
            QA.verify_run(run, root, require_frozen=False)
    finally:
        path.write_bytes(before)


def test_actual_console_latex_renders_all_terms_and_estimates_in_native_dom_reader(
        actual_source_results):
    _, run, _, proof = actual_source_results
    controller = (ROOT / 'desktop/scripts/verify_windows_ui.mjs').as_uri()
    require_base = (ROOT / 'web/package.json').as_uri()
    js = (f'import {{renderedTableRows}} from {json.dumps(controller)}; '
          "import {createRequire} from 'node:module'; import {readFileSync} from 'node:fs'; "
          f'const require=createRequire({json.dumps(require_base)}); '
          "const {JSDOM}=require('jsdom'); const katex=require('katex'); "
          "const sources=JSON.parse(readFileSync(0,'utf8')); "
          "const result=sources.map(source=> { const dom=new JSDOM('<div id=output></div>'); "
          "const node=dom.window.document.getElementById('output'); "
          "node.innerHTML=katex.renderToString(source,{throwOnError:true}); "
          "return renderedTableRows(node); }); console.log(JSON.stringify(result));")
    result = subprocess.run(['node', '--input-type=module', '-e', js], check=True,
                            input=json.dumps([item['latex_math'] for item in run['outputs']]),
                            capture_output=True, text=True, timeout=30)
    rendered = json.loads(result.stdout)
    for name, rows in zip(QA.NAMES, rendered, strict=True):
        expected = proof['tables'][name]['rows']
        assert len(rows) == len(expected) == 4
        for row, (term, estimate) in zip(rows, expected, strict=True):
            assert len(row) == 3 and row[1] == term
            assert abs(float(row[2]) - estimate) <= 0.000051


def test_explicit_controller_loading_works_with_isolated_python():
    script = ROOT / 'scripts/verify_regularized_all_family_runtime.py'
    code = (f'import importlib.util; s=importlib.util.spec_from_file_location("qa",{str(script)!r}); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            'w=m.load_controller(m.ROOT/"desktop/scripts/verify_windows_runtime.py"); '
            'p=m.load_controller(m.ROOT/"scripts/verify_bai_perron_runtime.py"); '
            'assert w.InstalledRuntime and p.OwnedRuntime; print("ISOLATED_CONTROLLERS_OK")')
    result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True,
                            text=True, check=True, timeout=30)
    assert result.stdout.strip() == 'ISOLATED_CONTROLLERS_OK'


def test_literal_node_ui_command_executes_all_seven_outputs_in_real_source_console(tmp_path):
    controller = (ROOT / 'desktop/scripts/verify_windows_ui.mjs').as_uri()
    example = ROOT / 'docs/examples/regularized_all_family.py'
    js = (f'import {{THREE_MODEL_CODE, regularizedCode}} from {json.dumps(controller)}; '
          "import {readFileSync} from 'node:fs'; "
          f'console.log(JSON.stringify(THREE_MODEL_CODE+"\\n"+regularizedCode(readFileSync({json.dumps(str(example))},"utf8"))));')
    compiled = subprocess.run(['node', '--input-type=module', '-e', js], check=True,
                              capture_output=True, text=True, timeout=30)
    code = json.loads(compiled.stdout).replace("assert getattr(sys, 'frozen', False)\n", '')
    assert example.read_text() in code
    assert len('exec(' + json.dumps(code) + ')') < 64000
    workspace = Workspace(tmp_path)
    session = ConsoleSession(workspace)
    try:
        run = session.execute(code, timeout_seconds=120)
    finally:
        session.close()
    assert run['status'] == 'ok', run['error']
    assert [output['type'] for output in run['outputs']] == ['model'] * 3 + ['table'] * 4
    assert 'WINDOWS_NATIVE_UI_THREE_MODELS_OK' in run['stdout']
    marker = 'WINDOWS_NATIVE_UI_REGULARIZED_FOUR_MODELS_OK '
    lines = [line for line in run['stdout'].splitlines() if line.startswith(marker)]
    assert len(lines) == 1 and len(run['stdout'].encode()) < 4096
    proof = json.loads(lines[0][len(marker):])
    assert proof['frozen'] is False
    assert proof['models'] == list(QA.NAMES)
    for name, expected in proof['files'].items():
        assert QA.digest(tmp_path / name) == expected['sha256']
    assert workspace.console_history()[0]['outputs'] == run['outputs']
