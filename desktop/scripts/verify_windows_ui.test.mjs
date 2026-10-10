// Controller proof validation only; actual native Windows receipts come from Actions.
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { REGULARIZED_MARKER, REGULARIZED_NAMES, regularizedCode, regularizedProofFromStdout,
  verifyRegularizedPresentation, SAVED_BINOMIAL_MARKER, SAVED_BINOMIAL_NAMES,
  savedBinomialCode, verifySavedBinomialPresentation, renderedNumberMatches,
  SAVED_BINOMIAL_END_MARKER, readOpenedLocalConsole, verifyOriginalLocalExecution,
  savedBinomialProofFromExecution, sanitizedEvaluationError } from './verify_windows_ui.mjs';

function fixture() {
  const filenames = ['regularized-all-family-models.json', 'regularized-all-family-data.csv',
    ...REGULARIZED_NAMES.flatMap(name => [`regularized-all-family-${name}.json`, `regularized-all-family-${name}.tex`])];
  const terms = ['Intercept', 'x', 'control', 'C(sector)[str:"red"]'];
  return { proof: { frozen: true, models: REGULARIZED_NAMES,
    files: Object.fromEntries(filenames.map(name => [name, { sha256: 'a'.repeat(64), bytes: 1000 }])),
    table_terms: Object.fromEntries(REGULARIZED_NAMES.map(name => [name, terms])),
    table_estimates: Object.fromEntries(REGULARIZED_NAMES.map(name => [name, [1.3, 0.7, 0.2, 0.3]])) },
  tables: REGULARIZED_NAMES.map(() => ({label: 'Table 4 rows · 2 columns',
    text: `Predictive term Estimate ${terms.join(' ')}`, visible: true,
    rows: terms.map((term, index) => [String(index), term, String([1.3, 0.7, 0.2, 0.3][index])])})) };
}

test('committed runnable example is embedded unchanged; complete state uses files and a small marker', async () => {
  const example = await readFile(new URL('../../docs/examples/regularized_all_family.py', import.meta.url), 'utf8');
  const code = regularizedCode(example);
  assert.ok(code.includes(example));
  assert.ok(code.includes("assert getattr(sys, 'frozen', False)"));
  assert.ok(code.includes("'frozen': getattr(sys, 'frozen', False)"));
  assert.ok(code.includes('WINDOWS_NATIVE_UI_REGULARIZED_FOUR_MODELS_OK '));
  assert.throws(() => regularizedCode('print("not the committed example")'), /committed runnable/);
});

test('all four full visible table presentations and exact saved file identities are required', () => {
  const {proof, tables} = fixture();
  const result = verifyRegularizedPresentation(proof, tables);
  assert.equal(Object.keys(result.saved_files).length, 10);
  assert.equal(result.rows_per_table, 4);
  assert.deepEqual(result.models, REGULARIZED_NAMES);
});

test('actual LF or Windows CRLF stdout must contain exactly one standalone success marker', () => {
  const {proof} = fixture();
  const line = REGULARIZED_MARKER + JSON.stringify(proof);
  assert.deepEqual(regularizedProofFromStdout(`ridge metrics\n${line}\n`), proof);
  assert.deepEqual(regularizedProofFromStdout(`ridge metrics\r\n${line}\r\n`), proof);
  assert.equal(regularizedProofFromStdout(`exec(\"print('${line}')\")`), null);
  assert.equal(regularizedProofFromStdout(`${line}\n${line}\n`), null);
});

for (const damage of ['source_worker', 'missing_model', 'missing_file', 'malformed_hash',
  'oversize_file', 'missing_table', 'invisible_table', 'truncated_rows', 'missing_term']) {
  test(`a ${damage} receipt cannot satisfy the native four-family gate`, () => {
    const {proof, tables} = fixture();
    if (damage === 'source_worker') proof.frozen = false;
    if (damage === 'missing_model') proof.models = proof.models.slice(0, 3);
    if (damage === 'missing_file') delete proof.files['regularized-all-family-pls.json'];
    if (damage === 'malformed_hash') proof.files['regularized-all-family-ridge.json'].sha256 = 'short';
    if (damage === 'oversize_file') proof.files['regularized-all-family-models.json'].bytes = 2 * 1024 * 1024 + 1;
    if (damage === 'missing_table') tables.pop();
    if (damage === 'invisible_table') tables[0].visible = false;
    if (damage === 'truncated_rows') tables[0].label = 'Table 3 rows · 2 columns';
    if (damage === 'missing_term') tables[0].text = tables[0].text.replace('control', '');
    assert.throws(() => verifyRegularizedPresentation(proof, tables));
  });
}

test('a visible table with different numeric estimates cannot satisfy the UI gate', () => {
  const {proof, tables} = fixture();
  tables[0].rows[0][2] = '9.9999';
  assert.throws(() => verifyRegularizedPresentation(proof, tables), /ridge table/);
});

function savedFixture() {
  const source = 'b'.repeat(40);
  const filenames = ['saved-binomial-suest-models.json', 'saved-binomial-suest-data.parquet',
    ...SAVED_BINOMIAL_NAMES.map(name => `saved-binomial-suest-${name}.tex`)];
  const rows = [['[a]', null, null, null, null, null, null],
    ...['Intercept', 'x', 'z'].map(term => [term, 1.234567, .12345, 10.00001, .00004, .9, 1.5]),
    ['[b]', null, null, null, null, null, null],
    ...['Intercept', 'x'].map(term => [term, 1.234567, .12345, 10.00001, .00004, .9, 1.5])];
  const proof = {frozen: true, fit_disabled_replay: false, source_sha: source,
    oracle_sha256: 'a'.repeat(64), cases: SAVED_BINOMIAL_NAMES, component_models: 16,
    joint_systems: 8, sample_sizes: [218, 240], union_positions: Array.from({length: 240}, (_, i) => i),
    files: Object.fromEntries(filenames.map(name => [name, {sha256: 'c'.repeat(64), bytes: 1000}])),
    table_rows: SAVED_BINOMIAL_NAMES.map(() => structuredClone(rows))};
  const tables = SAVED_BINOMIAL_NAMES.map(() => ({label: 'Table 7 rows · 7 columns', visible: true,
    rows: rows.map((row, index) => [String(index), ...row.map(value => value === null ? '—'
      : typeof value === 'number' ? (value !== 0 && Math.abs(value) < 1e-4
        ? value.toExponential(4) : value.toFixed(4)) : value)])}));
  return {source, proof, tables};
}

test('saved-binomial native code embeds unchanged fixture/oracle and requires real frozen execution', async () => {
  const example = await readFile(new URL('../../examples/saved_binomial_suest_acceptance.py', import.meta.url), 'utf8');
  const oracle = await readFile(new URL('../../examples/saved_binomial_suest.py', import.meta.url), 'utf8');
  const code = savedBinomialCode(example, oracle, 'b'.repeat(40));
  const nested = JSON.parse(code.trim().slice(5, -1));
  assert.ok(nested.startsWith(example));
  assert.ok(nested.includes(JSON.stringify(oracle)));
  assert.ok(nested.includes('emit=saved_native_emit'));
  assert.ok(!nested.includes('require_frozen=False'));
  assert.throws(() => savedBinomialCode(example, oracle, 'missing'), /Exact source/);
});

test('eight saved-binomial complete visible inference tables and exact source/file identities are required', () => {
  const {source, proof, tables} = savedFixture();
  assert.equal(verifySavedBinomialPresentation(proof, tables, source).joint_systems, 8);
  const marker = SAVED_BINOMIAL_MARKER + JSON.stringify(proof);
  assert.deepEqual(regularizedProofFromStdout(marker, SAVED_BINOMIAL_MARKER), proof);
  for (const damage of ['unfrozen', 'wrong_source', 'missing_file', 'wrong_union', 'truncated',
    'invisible', 'wrong_term', 'wrong_standard_error', 'wrong_p_value', 'filled_equation_header']) {
    const damaged = savedFixture();
    if (damage === 'unfrozen') damaged.proof.frozen = false;
    if (damage === 'wrong_source') damaged.proof.source_sha = 'd'.repeat(40);
    if (damage === 'missing_file') delete damaged.proof.files[Object.keys(damaged.proof.files)[0]];
    if (damage === 'wrong_union') damaged.proof.union_positions.pop();
    if (damage === 'truncated') damaged.tables[0].rows.pop();
    if (damage === 'invisible') damaged.tables[7].visible = false;
    if (damage === 'wrong_term') damaged.tables[0].rows[1][1] = 'other';
    if (damage === 'wrong_standard_error') damaged.tables[0].rows[1][3] = '.9';
    if (damage === 'wrong_p_value') damaged.tables[0].rows[1][5] = '.5';
    if (damage === 'filled_equation_header') damaged.tables[0].rows[0][2] = '0';
    assert.throws(() => verifySavedBinomialPresentation(damaged.proof, damaged.tables, source), damage);
  }
});

test('actual KaTeX scientific p-values preserve displayed precision and refuse zero substitution', () => {
  assert.ok(renderedNumberMatches('3.9607×10−9', 3.9607486e-9));
  assert.ok(renderedNumberMatches('7.3101×10−5', 7.3100917e-5));
  assert.ok(renderedNumberMatches('−0.1235', -.123456));
  assert.ok(renderedNumberMatches('4.0000×10−5', .00004));
  assert.ok(renderedNumberMatches('0.0000', 0));
  assert.ok(!renderedNumberMatches('0.0000', 3.9607486e-9));
  assert.ok(!renderedNumberMatches('0.0000×10−9', 3.9607486e-9));
  assert.ok(!renderedNumberMatches('3.9607×10−8', 3.9607486e-9));
  assert.ok(!renderedNumberMatches('3.9607×10−9', 3.9613e-9));
  for (const value of ['Infinity', 'NaN', '<.001', '123junk', '1e999']) {
    assert.ok(!renderedNumberMatches(value, .001));
  }
});

function executionFixture() {
  const {proof} = savedFixture();
  const project = 'a'.repeat(32), command = 'exec("the exact original fixture")';
  const before = 'metrics\n'.repeat(350) + 'WINDOWS_NATIVE_UI_THREE_MODELS_OK\n';
  const after = SAVED_BINOMIAL_MARKER + JSON.stringify(proof) + '\n' + SAVED_BINOMIAL_END_MARKER + '\n';
  const outputs = ['ols', 'logit', 'poisson'].map(estimator => ({type: 'model', data: {spec: {estimator}}}));
  outputs.push(...proof.table_rows.map(rows => ({type: 'table', data: {
    columns: ['Term', 'Estimate', 'SE', 'z', 'p', 'Low', 'High'], rows,
    total_rows: 7, total_columns: 7}})));
  const record = {id: 'original-id', code: command, status: 'ok', error: null,
    created_at: '2026-10-09T00:00:00Z', stdout: before + after, outputs,
    events: [...[0, 1, 2].map(index => ({type: 'output', index})), {type: 'stdout', text: before},
      ...Array.from({length: 8}, (_, index) => ({type: 'output', index: index + 3})),
      {type: 'stdout', text: after}]};
  return {project, command, proof, snapshot: {project_id: project, project_name: 'Owned project',
    status: {running: false}, history: [record]}};
}

test('truncated visible terminal prefix is separate from full original execution proof', () => {
  const {project, command, proof, snapshot} = executionFixture();
  const original = verifyOriginalLocalExecution(snapshot, command, project);
  assert.ok(original.record.stdout.length > 7488);
  const visible = original.record.stdout.slice(0, 7487) + '…';
  assert.ok(visible.includes(SAVED_BINOMIAL_MARKER));
  assert.throws(() => regularizedProofFromStdout(visible, SAVED_BINOMIAL_MARKER));
  assert.deepEqual(savedBinomialProofFromExecution(original.record), proof);
  const reopened = structuredClone(snapshot);
  // Durable JSON key ordering is not an execution change.
  reopened.history[0].outputs[0].data = {spec: {estimator: 'ols'}};
  assert.deepEqual(verifyOriginalLocalExecution(reopened, command, project, original), original);
});

test('combined three-model, four-family and saved-binomial output order is required', () => {
  const {project, command, proof, snapshot} = executionFixture();
  const {proof: prior} = fixture();
  const record = snapshot.history[0];
  record.outputs.splice(3, 0, ...REGULARIZED_NAMES.map(name => ({type: 'table', data: {
    columns: ['Predictive term', 'Estimate'], total_rows: 4, total_columns: 2,
    rows: prior.table_terms[name].map((term, index) => [term, prior.table_estimates[name][index]])}})));
  const first = record.events[3], last = record.events.at(-1);
  const regularized = {type: 'stdout', text: REGULARIZED_MARKER + JSON.stringify(prior) + '\n'};
  record.events = [...[0, 1, 2].map(index => ({type: 'output', index})), first,
    ...[3, 4, 5, 6].map(index => ({type: 'output', index})), regularized,
    ...Array.from({length: 8}, (_, index) => ({type: 'output', index: index + 7})), last];
  record.stdout = first.text + regularized.text + last.text;
  assert.deepEqual(savedBinomialProofFromExecution(
    verifyOriginalLocalExecution(snapshot, command, project).record, true), proof);
  record.outputs[3].data.rows[0][1] = 9;
  assert.throws(() => savedBinomialProofFromExecution(record, true), /persisted original table/);
});

test('evaluation diagnostics retain only constant guards and bounded exception names', () => {
  assert.match(sanitizedEvaluationError({exception: {className: 'Error',
    description: 'Error: The requested project is not the unique active local project.\nsecret stack'}}), /unique active/);
  assert.match(sanitizedEvaluationError({exception: {className: 'Error',
    description: 'Error: Local readback failed (403).\nprivate request'}}), /403/);
  assert.equal(sanitizedEvaluationError({exception: {className: 'SyntaxError',
    description: 'SyntaxError: token=secret; exec(full private code)'}}), 'Installed UI evaluation failed (SyntaxError).');
  assert.equal(sanitizedEvaluationError({exception: {className: 'secret-token/' + 'a'.repeat(500),
    description: 'private response body'}}), 'Installed UI evaluation failed (Error).');
});

for (const damage of ['wrong_project', 'extra_execution', 'wrong_command', 'failed_status',
  'missing_output', 'altered_event_text', 'reordered_events', 'missing_events',
  'altered_table_data', 'missing_end_marker', 'truncated_full_stdout', 'duplicate_end_marker',
  'end_marker_before_tables']) {
  test(`${damage} cannot satisfy complete original execution readback`, () => {
    const {project, command, snapshot} = executionFixture();
    const record = snapshot.history[0];
    if (damage === 'wrong_project') snapshot.project_id = 'b'.repeat(32);
    if (damage === 'extra_execution') snapshot.history.push(structuredClone(record));
    if (damage === 'wrong_command') record.code += '; print("rerun")';
    if (damage === 'failed_status') {record.status = 'error'; record.error = {message: 'failed'};}
    if (damage === 'missing_output') record.outputs.pop();
    if (damage === 'altered_event_text') record.events.at(-1).text += 'changed';
    if (damage === 'reordered_events') [record.events[0], record.events[1]] = [record.events[1], record.events[0]];
    if (damage === 'missing_events') delete record.events;
    if (damage === 'altered_table_data') record.outputs[3].data.rows[1][2] = .99;
    if (['missing_end_marker', 'truncated_full_stdout', 'duplicate_end_marker'].includes(damage)) {
      const text = record.events.at(-1).text;
      record.events.at(-1).text = damage === 'missing_end_marker' ? text.replace(SAVED_BINOMIAL_END_MARKER + '\n', '')
        : damage === 'truncated_full_stdout' ? text.slice(0, 400)
        : text + SAVED_BINOMIAL_END_MARKER + '\n';
      record.stdout = record.events.filter(event => event.type === 'stdout').map(event => event.text).join('');
    }
    if (damage === 'end_marker_before_tables') {
      record.events[3].text += SAVED_BINOMIAL_END_MARKER + '\n';
      record.events.at(-1).text = record.events.at(-1).text.replace(SAVED_BINOMIAL_END_MARKER + '\n', '');
      record.stdout = record.events.filter(event => event.type === 'stdout').map(event => event.text).join('');
    }
    assert.throws(() => savedBinomialProofFromExecution(
      verifyOriginalLocalExecution(snapshot, command, project).record), damage);
  });
}

for (const field of ['id', 'code', 'stdout', 'outputs', 'events', 'status', 'created_at']) {
  test(`cold reopening refuses an altered original ${field}`, () => {
    const {project, command, snapshot} = executionFixture();
    const original = verifyOriginalLocalExecution(snapshot, command, project);
    const reopened = structuredClone(snapshot);
    if (field === 'outputs') reopened.history[0].outputs[3].data.rows[1][2] += .1;
    else if (field === 'events') reopened.history[0].events.at(-1).text += 'changed';
    else reopened.history[0][field] += '-changed';
    assert.throws(() => verifyOriginalLocalExecution(reopened, command, project, original));
  });
}

function transportFixture(damage) {
  const {project, snapshot} = executionFixture();
  const requests = [];
  let statuses = 0;
  const fetcher = async (path, options) => {
    requests.push({path, options});
    let body;
    if (path === '/api/desktop/session') body = {environment: 'desktop', persistent: true, token: 'desktop-secret'};
    else if (path === '/api/desktop/status') {
      statuses++;
      assert.equal(options.headers['X-OpenEcon-Token'], 'desktop-secret');
      body = {project_id: damage === 'changed_project' && statuses === 2 ? 'b'.repeat(32) : project,
        persistent: true, execution_mode: 'persistent'};
    } else if (path === '/api/desktop/local-projects') {
      assert.equal(options.headers['X-OpenEcon-Token'], 'desktop-secret');
      body = {projects: [{id: damage === 'incorrect_project' ? 'b'.repeat(32) : project, name: 'Owned project'}]};
      if (damage === 'ambiguous_project') body.projects.push({id: 'b'.repeat(32), name: 'Owned project'});
    } else if (path.endsWith('/session')) body = {environment: 'local', persistent: true, token: 'project-secret'};
    else if (path.endsWith('/console')) {
      assert.equal(options.headers['X-OpenEcon-Token'], 'project-secret');
      body = {history: snapshot.history, status: snapshot.status};
    } else throw new Error(`Unexpected path: ${path}`);
    return {ok: damage !== 'denied', status: damage === 'denied' ? 403 : 200, json: async () => body};
  };
  return {project, snapshot, requests, fetcher};
}

test('active project reader uses only read-only project routes and returns no credentials', async () => {
  const {project, snapshot, requests, fetcher} = transportFixture();
  const result = await readOpenedLocalConsole('Owned project', project, fetcher);
  assert.deepEqual(result, snapshot);
  assert.ok(!JSON.stringify(result).includes('secret'));
  assert.ok(requests.every(request => request.options.method === undefined));
  assert.deepEqual(requests.map(request => request.path), ['/api/desktop/session', '/api/desktop/status',
    '/api/desktop/local-projects', `/api/desktop/projects/${project}/workspace/session`,
    `/api/desktop/projects/${project}/workspace/console`, '/api/desktop/status']);
});

for (const damage of ['incorrect_project', 'ambiguous_project', 'changed_project', 'denied']) {
  test(`local session readback refuses ${damage}`, async () => {
    const {project, fetcher} = transportFixture(damage);
    await assert.rejects(readOpenedLocalConsole('Owned project', project, fetcher));
  });
}

// Prebuild controller tests need only Node. The explicit interpreter activates
// an additional real source worker/API fixture after runtime dependencies exist.
if (process.env.OPENECON_TEST_PYTHON) {
  test('real local desktop worker persists full stdout and ordered outputs across cold readback',
    {timeout: 120000}, async () => {
      const directory = await mkdtemp(join(tmpdir(), 'openecon-original-execution-'));
      const resultFile = join(directory, 'readback.json');
      const script = join(directory, 'fixture.py');
      const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
      const command = "model = oe.ols(data=oe.example(), y='wage', x=['education', 'experience'], covariance='HC3')\nprint('prefix-' + 'x' * 9000)\ndisplay(model)\ndisplay(pd.DataFrame({'Term': ['x'], 'Estimate': [1.25]}))\nprint('REAL_LOCAL_WORKER_END')";
      await writeFile(script, `import json
from pathlib import Path
from fastapi.testclient import TestClient
from openecon.desktop_runtime import create_desktop_app

def main():
    root = Path(${JSON.stringify(directory)}) / 'desktop'
    command = ${JSON.stringify(command)}
    def read(client):
        desktop = client.get('/api/desktop/session').json()
        assert desktop['environment'] == 'desktop' and desktop['persistent'] is True
        headers = {'X-OpenEcon-Token': desktop['token']}
        active = client.get('/api/desktop/status', headers=headers).json()
        catalog = client.get('/api/desktop/local-projects', headers=headers).json()
        project = catalog['projects'][0]
        assert project['id'] == active['project_id']
        prefix = '/api/desktop/projects/' + project['id'] + '/workspace'
        session = client.get(prefix + '/session').json()
        assert session['environment'] == 'local' and session['persistent'] is True
        state = client.get(prefix + '/console', headers={'X-OpenEcon-Token': session['token']}).json()
        return {'project_id': project['id'], 'project_name': project['name'], 'status': {'running': state['status']['running']}, 'history': [{key: record[key] for key in ['id', 'code', 'status', 'error', 'stdout', 'outputs', 'events', 'created_at']} for record in state['history']]}
    with TestClient(create_desktop_app(root)) as client:
        token = client.get('/api/desktop/session').json()['token']
        headers = {'X-OpenEcon-Token': token}
        project = client.post('/api/desktop/local-projects', headers=headers, json={'name': 'Real source fixture'}).json()
        assert client.post('/api/desktop/local-projects/' + project['id'] + '/open', headers=headers).status_code == 200
        prefix = '/api/desktop/projects/' + project['id'] + '/workspace'
        session = client.get(prefix + '/session').json()
        run = client.post(prefix + '/console/execute', headers={'X-OpenEcon-Token': session['token']}, json={'code': command}).json()
        assert run['status'] == 'ok', run
        first = read(client)
    with TestClient(create_desktop_app(root)) as client:
        token = client.get('/api/desktop/session').json()['token']
        assert client.post('/api/desktop/local-projects/' + project['id'] + '/open', headers={'X-OpenEcon-Token': token}).status_code == 200
        second = read(client)
        assert client.app.state.desktop_projects.active_app.state.console._process is None
    Path(${JSON.stringify(resultFile)}).write_text(json.dumps({'command': command, 'first': first, 'second': second}), encoding='utf-8')

if __name__ == '__main__':
    main()
`);
      try {
        await new Promise((resolveRun, reject) => {
          const child = spawn(process.env.OPENECON_TEST_PYTHON, [script], {cwd: root,
            env: {...process.env, PYTHONPATH: [join(root, 'src'), join(root, 'packages/openecon-charts/src')].join(process.platform === 'win32' ? ';' : ':')},
            stdio: ['ignore', 'pipe', 'pipe']});
          let log = '';
          child.stdout.on('data', chunk => {log += chunk;});
          child.stderr.on('data', chunk => {log += chunk;});
          child.once('error', reject);
          child.once('exit', code => code === 0 ? resolveRun() : reject(new Error(`Real worker failed (${code}): ${log}`)));
        });
        const result = JSON.parse(await readFile(resultFile, 'utf8'));
        const original = verifyOriginalLocalExecution(result.first, result.command, result.first.project_id);
        assert.ok(original.record.stdout.length > 8000 && original.record.stdout.endsWith('REAL_LOCAL_WORKER_END\n'));
        assert.equal(original.record.outputs.length, 2);
        assert.equal(original.record.outputs[0].type, 'model');
        assert.equal(original.record.outputs[0].data.spec.estimator, 'ols');
        assert.equal(original.record.outputs[1].type, 'table');
        assert.deepEqual(original.record.outputs[1].data.columns, ['Term', 'Estimate']);
        assert.deepEqual(original.record.outputs[1].data.rows, [['x', 1.25]]);
        assert.equal(original.record.outputs[1].data.total_rows, 1);
        assert.equal(original.record.outputs[1].data.total_columns, 2);
        verifyOriginalLocalExecution(result.second, result.command, result.first.project_id, original);
      } finally {
        await rm(directory, {recursive: true, force: true});
      }
    });
}
