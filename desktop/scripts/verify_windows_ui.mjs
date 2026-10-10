// Drive only the disposable runner's installed WebView2 through its loopback CDP.
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

export const REGULARIZED_MARKER = 'WINDOWS_NATIVE_UI_REGULARIZED_FOUR_MODELS_OK ';
export const REGULARIZED_NAMES = ['ridge', 'lasso', 'elasticnet', 'pls'];
export const SAVED_BINOMIAL_MARKER = 'SAVED_BINOMIAL_SUEST_EIGHT_OK ';
export const SAVED_BINOMIAL_END_MARKER = 'SAVED_BINOMIAL_SUEST_ORIGINAL_EXECUTION_END';
export const SAVED_BINOMIAL_NAMES = ['cloglog_offset', 'cloglog_frequency_cluster',
  'fractional_logit_corners', 'fractional_probit_corners', 'fractional_logit_frequency',
  'fractional_probit_probability_cluster', 'fractional_logit_analytic',
  'fractional_probit_analytic_cluster'];

export function renderedNumberMatches(cell, expected) {
  if (typeof expected !== 'number' || !Number.isFinite(expected)) return false;
  const text = String(cell).trim().replaceAll('−', '-').replaceAll(' ', '');
  const match = /^([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:×10([+-]?\d+)|[eE]([+-]?\d+))?$/.exec(text);
  if (!match) return false;
  const exponent = Number(match[2] ?? match[3] ?? 0);
  const value = Number(match[1]) * 10 ** exponent;
  if (value === 0 && expected !== 0) return false;
  const decimals = match[1].split('.')[1]?.length ?? 0;
  const rounding = .50001 * 10 ** (exponent - decimals);
  const tolerance = Math.max(rounding, Number.EPSILON * Math.max(Math.abs(value), Math.abs(expected)) * 4);
  return Number.isFinite(value) && Number.isFinite(tolerance) && Math.abs(value - expected) <= tolerance;
}

export function savedBinomialCode(example, oracle, sourceSha) {
  if (!/^[0-9a-f]{40}$/.test(sourceSha) || !example.includes('def saved_binomial_acceptance(')
      || !oracle.includes('def expected_covariance(')) {
    throw new Error('Exact source identity and saved binomial fixture/oracle are required.');
  }
  const invocation = `\ndef saved_native_emit(joint):\n    display(pd.DataFrame(saved_binomial_table(joint), columns=['Term', 'Estimate', 'SE', 'z', 'p', 'Low', 'High']))\nsaved_binomial_acceptance(${JSON.stringify(oracle)}, ${JSON.stringify(sourceSha)}, emit=saved_native_emit)\nprint('${SAVED_BINOMIAL_END_MARKER}')\n`;
  // The unchanged fixture's future import remains first in its own namespace.
  return `\nexec(${JSON.stringify(example + invocation)})\n`;
}

export function verifySavedBinomialPresentation(proof, tables, sourceSha) {
  const files = ['saved-binomial-suest-models.json', 'saved-binomial-suest-data.parquet',
    ...SAVED_BINOMIAL_NAMES.map(name => `saved-binomial-suest-${name}.tex`)];
  if (proof?.frozen !== true || proof.fit_disabled_replay !== false
      || !/^[0-9a-f]{40}$/.test(sourceSha) || proof.source_sha !== sourceSha
      || !/^[0-9a-f]{64}$/.test(proof.oracle_sha256)
      || JSON.stringify(proof.cases) !== JSON.stringify(SAVED_BINOMIAL_NAMES)
      || proof.component_models !== 16 || proof.joint_systems !== 8
      || JSON.stringify(proof.sample_sizes) !== '[218,240]'
      || JSON.stringify(proof.union_positions) !== JSON.stringify(Array.from({length: 240}, (_, i) => i))
      || JSON.stringify(Object.keys(proof.files ?? {}).sort()) !== JSON.stringify(files.sort())
      || Object.values(proof.files).some(file => !/^[0-9a-f]{64}$/.test(file.sha256)
        || !Number.isInteger(file.bytes) || file.bytes < 1 || file.bytes > 2 * 1024 * 1024)
      || tables.length !== 8 || !Array.isArray(proof.table_rows) || proof.table_rows.length !== 8) {
    throw new Error('Complete frozen saved-binomial identity, state and eight table outputs are required.');
  }
  for (const [index, table] of tables.entries()) {
    const expected = proof.table_rows[index];
    if (!Array.isArray(expected) || expected.length !== 7 || expected.some(row => row.length !== 7)
        || !table.visible || !String(table.label).includes('7 rows')
        || !String(table.label).includes('7 columns') || !Array.isArray(table.rows)
        || table.rows.length !== 7 || table.rows.some((row, position) => row.length !== 8
          || row[1] !== expected[position][0] || row.slice(2).some((cell, column) => {
            const value = expected[position][column + 1];
            return value === null ? !['—', '-', '', 'NaN', 'nan'].includes(cell)
              : !renderedNumberMatches(cell, value);
          }))) {
      throw new Error(`The actual ${SAVED_BINOMIAL_NAMES[index]} full coefficient/inference table is incomplete.`);
    }
  }
  return {cases: SAVED_BINOMIAL_NAMES, saved_files: proof.files, tables, rows_per_table: 7,
    component_models: 16, joint_systems: 8, source_sha: sourceSha, oracle_sha256: proof.oracle_sha256};
}

export const THREE_MODEL_CODE = `import json, math
from pathlib import Path
import openecon as oe
from openecon.models import ResultBundle
x = [float(i // 2) - 14.5 for i in range(60)]
b = [float(i % 2) for i in range(60)]
c = [1.0 + 2.0 * (i % 2) for i in range(60)]
y = [3.0 + 1.5 * v + 0.2 * math.sin(i) for i, v in enumerate(x)]
Path('windows-ui.csv').write_text('x,y,b,c\\n' + ''.join(f'{a},{d},{e},{f}\\n' for a,d,e,f in zip(x,y,b,c)), encoding='utf-8')
data = oe.read('windows-ui.csv')
ols = oe.ols(data=data, y='y', x=['x'], covariance='HC3')
logit = oe.logit(data=data, y='b', x=['x'])
poisson = oe.poisson(data=data, y='c', x=['x'])
sx = sum(v*v for v in x)
mx, my = sum(x)/60, sum(y)/60
slope = sum((a-mx)*(d-my) for a,d in zip(x,y))/sum((a-mx)**2 for a in x)
assert math.isclose(ols.coefficients[0].estimate, my-slope*mx, abs_tol=1e-9)
assert math.isclose(ols.coefficients[1].estimate, slope, abs_tol=1e-9)
for model, intercept, factor in [(logit, 0.0, 4.0), (poisson, math.log(2), 0.5)]:
    assert model.nobs == 60 and model.sample_positions == list(range(60))
    assert math.isclose(model.coefficients[0].estimate, intercept, abs_tol=1e-7)
    assert math.isclose(model.coefficients[1].estimate, 0.0, abs_tol=1e-7)
    expected = [[factor/60, 0.0], [0.0, factor/sx]]
    assert all(math.isclose(model.covariance_matrix[i][j], expected[i][j], rel_tol=1e-6, abs_tol=1e-9) for i in range(2) for j in range(2))
for name, model in [('ols',ols),('logit',logit),('poisson',poisson)]:
    Path(f'windows-ui-{name}.json').write_text(model.model_dump_json(), encoding='utf-8')
    saved = ResultBundle.model_validate_json(Path(f'windows-ui-{name}.json').read_text(encoding='utf-8'))
    assert saved.model_dump(mode='json') == model.model_dump(mode='json')
    globals()['display'](model)
print('WINDOWS_NATIVE_UI_THREE_MODELS_OK')`;

export function regularizedCode(example) {
  if (!example.includes('REGULARIZED_ALL_FAMILY_STATE_REPLAY_OK')) {
    throw new Error('The exact committed runnable example is required.');
  }
  return `import sys, hashlib, json
from pathlib import Path
assert getattr(sys, 'frozen', False)
` + example + `
models = {name: result.model_dump(mode='json') for name, result in results.items()}
predictions = {name: oe.regularized_predict(result, data).tolist() for name, result in results.items()}
tables = {name: {'columns': list(oe.regularized_table(result).columns), 'rows': oe.regularized_table(result).values.tolist()} for name, result in results.items()}
Path('regularized-all-family-models.json').write_text(json.dumps({'models': models, 'predictions': predictions, 'tables': tables}, sort_keys=True, allow_nan=False), encoding='utf-8')
data.to_csv('regularized-all-family-data.csv', index=False)
for name, result in results.items():
    path = Path('regularized-all-family-' + name + '.json')
    path.write_text(result.model_dump_json(), encoding='utf-8')
    restored = ResultBundle.model_validate_json(path.read_text(encoding='utf-8'))
    assert restored.model_dump(mode='json') == models[name]
    assert oe.regularized_predict(restored, data).tolist() == predictions[name]
    Path('regularized-all-family-' + name + '.tex').write_text(str(oe.regularized_table(restored).to_latex(index=False)), encoding='utf-8')
files = {path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size} for path in Path('.').glob('regularized-all-family-*') if path.is_file()}
print('${REGULARIZED_MARKER}' + json.dumps({'frozen': getattr(sys, 'frozen', False), 'models': list(results), 'files': files, 'table_terms': {name: [row[0] for row in table['rows']] for name, table in tables.items()}, 'table_estimates': {name: [row[1] for row in table['rows']] for name, table in tables.items()}}, sort_keys=True, allow_nan=False))`;
}

export function verifyRegularizedPresentation(proof, tables) {
  if (proof?.frozen !== true || JSON.stringify(proof.models) !== JSON.stringify(REGULARIZED_NAMES)) {
    throw new Error('The actual frozen four-family result is missing.');
  }
  const files = ['regularized-all-family-models.json', 'regularized-all-family-data.csv',
    ...REGULARIZED_NAMES.flatMap(name => [`regularized-all-family-${name}.json`, `regularized-all-family-${name}.tex`])];
  if (JSON.stringify(Object.keys(proof.files ?? {}).sort()) !== JSON.stringify(files.sort())
      || Object.values(proof.files).some(file => !/^[0-9a-f]{64}$/.test(file.sha256)
        || !Number.isInteger(file.bytes) || file.bytes < 1 || file.bytes > 2 * 1024 * 1024)
      || tables.length !== 4) {
    throw new Error('Complete saved four-family files and four real table outputs are required.');
  }
  for (const [index, table] of tables.entries()) {
    const terms = proof.table_terms?.[REGULARIZED_NAMES[index]];
    const estimates = proof.table_estimates?.[REGULARIZED_NAMES[index]];
    const text = String(table.text).replace(/\s+/g, ' ');
    const label = String(table.label).replace(/\s+/g, ' ');
    if (!Array.isArray(terms) || terms.length !== 4 || !table.visible
        || !label.includes('4 rows') || !label.includes('2 columns')
        || !text.includes('Predictive term') || !text.includes('Estimate')
        || terms.some(term => !text.includes(term))
        || !Array.isArray(estimates) || estimates.length !== 4
        || !Array.isArray(table.rows) || table.rows.length !== 4
        || table.rows.some((row, rowIndex) => row.length !== 3 || row[1] !== terms[rowIndex]
          || !Number.isFinite(Number(row[2])) || !Number.isFinite(estimates[rowIndex])
          || Math.abs(Number(row[2]) - estimates[rowIndex]) > 0.000051)) {
      throw new Error(`The actual ${REGULARIZED_NAMES[index]} table is incomplete or invisible.`);
    }
  }
  return { models: REGULARIZED_NAMES, saved_files: proof.files, tables, rows_per_table: 4,
    full_model_json_bytes: proof.files['regularized-all-family-models.json'].bytes };
}

export function renderedTableRows(presentation) {
  const mathRows = [...presentation.querySelectorAll('math mtable > mtr')]
    .map(row => [...row.children].map(cell => cell.textContent.replace(/\s+/g, ' ').trim()));
  return mathRows.length ? mathRows.filter(row => row[1] !== 'Predictive term')
    : [...presentation.querySelectorAll('tbody tr')].map(row => [...row.children].map(cell => cell.textContent.replace(/\s+/g, ' ').trim()));
}

export function regularizedProofFromStdout(stdout, marker = REGULARIZED_MARKER) {
  const lines = String(stdout).split(/\r?\n/).filter(line => line.startsWith(marker));
  if (lines.length !== 1) return null;
  return JSON.parse(lines[0].slice(marker.length));
}

// This function is serialized into the trusted native page. Tokens never leave
// its closure, and every operation reads the project already opened by the UI.
export async function readOpenedLocalConsole(projectName, projectId = null, fetcher = fetch) {
  async function read(path, token) {
    const response = await fetcher(path, {headers: token ? {'X-OpenEcon-Token': token} : {},
      signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error(`Local readback failed (${response.status}).`);
    return response.json();
  }
  const desktop = await read('/api/desktop/session');
  if (desktop.environment !== 'desktop' || desktop.persistent !== true || !desktop.token) {
    throw new Error('The persistent local desktop session is missing.');
  }
  const active = await read('/api/desktop/status', desktop.token);
  const catalog = await read('/api/desktop/local-projects', desktop.token);
  const projects = catalog.projects?.filter(project => project.name === projectName);
  if (!/^[0-9a-f]{32}$/.test(active.project_id ?? '') || active.persistent !== true
      || active.execution_mode !== 'persistent' || projects?.length !== 1
      || projects[0].id !== active.project_id || (projectId !== null && projectId !== active.project_id)) {
    throw new Error('The requested project is not the unique active local project.');
  }
  const prefix = `/api/desktop/projects/${active.project_id}/workspace`;
  const session = await read(prefix + '/session');
  if (session.environment !== 'local' || session.persistent !== true || !session.token) {
    throw new Error('The opened project local session is missing.');
  }
  const state = await read(prefix + '/console', session.token);
  const after = await read('/api/desktop/status', desktop.token);
  if (after.project_id !== active.project_id || !Array.isArray(state.history)) {
    throw new Error('The local project changed during original execution readback.');
  }
  return {project_id: active.project_id, project_name: projects[0].name,
    status: {running: state.status?.running},
    history: state.history.map(record => Object.fromEntries(
      ['id', 'code', 'status', 'error', 'stdout', 'outputs', 'events', 'created_at']
        .map(key => [key, record[key]])))};
}

export function verifyOriginalLocalExecution(readback, command, projectId, previous = null) {
  const record = readback?.history?.[0];
  if (readback?.project_id !== projectId || readback.status?.running !== false
      || readback.history.length !== 1 || typeof record?.id !== 'string' || !record.id
      || record.code !== command || record.status !== 'ok' || record.error !== null
      || typeof record.stdout !== 'string' || !Array.isArray(record.outputs)
      || record.outputs.length > 20 || !Array.isArray(record.events)
      || record.events.length > 2 * record.outputs.length + 1) {
    throw new Error('One successful original local execution with the exact command is required.');
  }
  const text = [];
  let next = 0, last;
  for (const event of record.events) {
    const keys = Object.keys(event).sort().join(',');
    if (event.type === 'stdout' && keys === 'text,type' && typeof event.text === 'string'
        && event.text && last !== 'stdout') text.push(event.text);
    else if (event.type === 'output' && keys === 'index,type' && event.index === next
        && next < record.outputs.length) next++;
    else throw new Error('The original execution ordered event timeline is invalid.');
    last = event.type;
  }
  if (next !== record.outputs.length || text.join('') !== record.stdout
      || Buffer.byteLength(record.stdout) > 70 * 1024) {
    throw new Error('The original execution stdout or ordered outputs are incomplete.');
  }
  const original = {project_id: projectId, project_name: readback.project_name, record};
  const canonical = value => JSON.stringify(value, function(key, item) {
    return item && typeof item === 'object' && !Array.isArray(item)
      ? Object.fromEntries(Object.keys(item).sort().map(name => [name, item[name]])) : item;
  });
  if (previous && canonical(previous) !== canonical(original)) {
    throw new Error('Cold readback differs from the saved original project execution.');
  }
  return original;
}

export function savedBinomialProofFromExecution(record, regularized = false) {
  const lines = record.stdout.split(/\r?\n/).filter(line => line !== '');
  if (lines.filter(line => line === SAVED_BINOMIAL_END_MARKER).length !== 1
      || lines.at(-1) !== SAVED_BINOMIAL_END_MARKER
      || lines.filter(line => line === 'WINDOWS_NATIVE_UI_THREE_MODELS_OK').length !== 1) {
    throw new Error('The complete original execution success and end markers are required.');
  }
  const proof = regularizedProofFromStdout(record.stdout, SAVED_BINOMIAL_MARKER);
  const prior = regularized ? regularizedProofFromStdout(record.stdout) : null;
  const offset = 3 + (regularized ? 4 : 0);
  if (!proof || (regularized && !prior) || record.outputs.length !== offset + 8
      || record.outputs.slice(0, 3).some((output, index) => output.type !== 'model'
        || output.data?.spec?.estimator !== ['ols', 'logit', 'poisson'][index])) {
    throw new Error('The original ordered model and table outputs are incomplete.');
  }
  for (const [index, output] of record.outputs.slice(3).entries()) {
    const saved = index >= (regularized ? 4 : 0);
    const expected = saved ? proof.table_rows?.[index - (regularized ? 4 : 0)]
      : prior.table_terms?.[REGULARIZED_NAMES[index]]?.map((term, row) =>
        [term, prior.table_estimates?.[REGULARIZED_NAMES[index]]?.[row]]);
    const columns = saved ? ['Term', 'Estimate', 'SE', 'z', 'p', 'Low', 'High']
      : ['Predictive term', 'Estimate'];
    if (output.type !== 'table' || JSON.stringify(output.data?.columns) !== JSON.stringify(columns)
        || JSON.stringify(output.data?.rows) !== JSON.stringify(expected)
        || output.data?.total_rows !== (saved ? 7 : 4)
        || output.data?.total_columns !== columns.length) {
      throw new Error('The persisted original table data differs from its complete proof.');
    }
  }
  let outputs = 0;
  for (const event of record.events) {
    if (event.type === 'output') outputs++;
    else for (const [marker, count] of [['WINDOWS_NATIVE_UI_THREE_MODELS_OK', 3],
      [REGULARIZED_MARKER, 7], [SAVED_BINOMIAL_MARKER, offset + 8],
      [SAVED_BINOMIAL_END_MARKER, offset + 8]]) {
      if (event.text.includes(marker) && outputs !== count) {
        throw new Error('The original success markers do not follow their ordered outputs.');
      }
    }
  }
  return proof;
}

export function sanitizedEvaluationError(details) {
  const name = /^[A-Za-z][A-Za-z0-9]{0,49}$/.test(details?.exception?.className ?? '')
    ? details.exception.className : 'Error';
  const first = String(details?.exception?.description ?? '').split('\n')[0].replace(/^Error: /, '');
  // Only controller-owned constant guards may cross the CDP exception boundary.
  const guards = ['The persistent local desktop session is missing.',
    'The requested project is not the unique active local project.',
    'The opened project local session is missing.',
    'The local project changed during original execution readback.'];
  return `Installed UI evaluation failed (${name})${guards.includes(first)
    || /^Local readback failed \([1-5][0-9]{2}\)\.$/.test(first) ? ': ' + first : '.'}`;
}

async function main() {
  const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, array) => {
    if (index % 2 === 0) pairs.push([value.replace(/^--/, ''), array[index + 1]]);
    return pairs;
  }, []));
  const port = Number(args.port);
  if (process.platform !== 'win32' || process.env.GITHUB_ACTIONS !== 'true'
      || !Number.isInteger(port) || port < 1024 || port > 65535
      || !['create', 'reopen'].includes(args.mode) || !args.output || !args['project-name']
      || (args['regularized-all-family'] !== undefined && !['true', 'false'].includes(args['regularized-all-family']))
      || (args['saved-binomial'] !== undefined && !['true', 'false'].includes(args['saved-binomial']))
      || (args['saved-binomial'] === 'true' && (!/^[0-9a-f]{40}$/.test(args['source-sha'] ?? '')
        || (args.mode === 'reopen' && !args['original-receipt'])))) {
    throw new Error('A disposable GitHub Windows runner and explicit UI arguments are required.');
  }
  const regularizedEnabled = args['regularized-all-family'] === 'true';
  const savedBinomialEnabled = args['saved-binomial'] === 'true';
  const receipt = { status: 'running', regularized_all_family_enabled: regularizedEnabled, mode: args.mode, project_name: args['project-name'],
    saved_binomial_enabled: savedBinomialEnabled,
    controller: 'Node loopback CDP; installed native WebView2', checks: {} };
  let socket;
  let nextId = 0;
  const pending = new Map();
  async function until(operation, timeout = 180000) {
    const deadline = Date.now() + timeout;
    let last;
    while (Date.now() < deadline) {
      try { const value = await operation(); if (value) return value; }
      catch (error) { last = error; }
      await delay(250);
    }
    throw new Error(last ? `UI condition timed out: ${last.message}` : 'UI condition timed out.');
  }
  function request(method, params = {}) {
    const id = ++nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`${method} timed out.`)); }, 300000);
      pending.set(id, { resolve, reject, timer });
      socket.send(JSON.stringify({ id, method, params }));
    });
  }
  async function evaluate(expression) {
    const result = await request('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(sanitizedEvaluationError(result.exceptionDetails));
    return result.result?.value;
  }
  async function click(selector) {
    const point = await until(() => evaluate(`(() => {
      const node = document.querySelector(${JSON.stringify(selector)});
      if (!node || node.disabled) return null;
      const rect = node.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return null;
      node.scrollIntoView({block: 'center'});
      const visible = node.getBoundingClientRect();
      return {x: visible.x + visible.width/2, y: visible.y + visible.height/2};
    })()`));
    await request('Input.dispatchMouseEvent', { type: 'mousePressed', ...point, button: 'left', clickCount: 1 });
    await request('Input.dispatchMouseEvent', { type: 'mouseReleased', ...point, button: 'left', clickCount: 1 });
  }
  async function fill(selector, text) {
    await until(() => evaluate(`(() => {
      const node = document.querySelector(${JSON.stringify(selector)});
      if (!node || node.disabled) return false;
      node.focus(); return true;
    })()`));
    await request('Input.insertText', { text });
  }
  let code = THREE_MODEL_CODE;

  if (regularizedEnabled) {
    const example = await readFile(new URL('../../docs/examples/regularized_all_family.py', import.meta.url), 'utf8');
    receipt.regularized_example_sha256 = createHash('sha256').update(example).digest('hex');
    code += '\n' + regularizedCode(example);
  }
  if (savedBinomialEnabled) {
    const example = await readFile(new URL('../../examples/saved_binomial_suest_acceptance.py', import.meta.url), 'utf8');
    const oracle = await readFile(new URL('../../examples/saved_binomial_suest.py', import.meta.url), 'utf8');
    receipt.saved_binomial_example_sha256 = createHash('sha256').update(example).digest('hex');
    receipt.saved_binomial_oracle_sha256 = createHash('sha256').update(oracle).digest('hex');
    code += savedBinomialCode(example, oracle, args['source-sha']);
  }
  const command = `exec(${JSON.stringify(code)})`;
  let originalReceipt = null;
  if (savedBinomialEnabled && args.mode === 'reopen') {
    originalReceipt = JSON.parse(await readFile(args['original-receipt'], 'utf8'));
    if (originalReceipt.status !== 'passed' || originalReceipt.mode !== 'create'
        || originalReceipt.saved_binomial_enabled !== true
        || originalReceipt.project_name !== args['project-name']
        || originalReceipt.saved_binomial?.source_sha !== args['source-sha']
        || !originalReceipt.original_execution) {
      throw new Error('A successful exact-source original create receipt is required for cold readback.');
    }
  }

  try {
    const target = await until(async () => {
      const response = await fetch(`http://127.0.0.1:${port}/json/list`, { signal: AbortSignal.timeout(5000) });
      if (!response.ok) return null;
      const targets = await response.json();
      return targets.find(row => {
        try { const url = new URL(row.url); return row.type === 'page' && url.protocol === 'http:' && url.hostname === '127.0.0.1'; }
        catch { return false; }
      });
    });
    const debuggerUrl = new URL(target.webSocketDebuggerUrl);
    if (debuggerUrl.hostname !== '127.0.0.1' || Number(debuggerUrl.port) !== port) throw new Error('Untrusted CDP endpoint.');
    socket = new WebSocket(debuggerUrl);
    await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', () => reject(new Error('Owned CDP socket failed.')), { once: true }); });
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      const entry = pending.get(message.id);
      if (!entry) return;
      pending.delete(message.id); clearTimeout(entry.timer);
      if (message.error) entry.reject(new Error(`CDP ${message.error.code}`)); else entry.resolve(message.result);
    });
    await request('Runtime.enable');
    await until(() => evaluate(`document.body.innerText.includes('Local projects')`));
    if (!await evaluate(`typeof window.__TAURI__?.core?.invoke === 'function'`)) throw new Error('The actual native bridge is missing.');
    // Invoke on the trusted local page; do not retain credentials or the descriptor.
    if (!await evaluate(`window.__TAURI__.core.invoke('desktop_info').then(() => true)`)) throw new Error('Trusted native IPC failed.');
    receipt.checks.actual_normal_native_window_and_trusted_ipc = true;
    receipt.desktop_ipc_verified = true;
    if (args.mode === 'create') {
      await fill('input[aria-label="Local project name"]', args['project-name']);
      await click('.local-project-form button');
    } else {
      const name = JSON.stringify(args['project-name']);
      const selector = await until(() => evaluate(`(() => {
        const cards = [...document.querySelectorAll('.team-project-open')];
        const index = cards.findIndex(node => node.querySelector('h2')?.textContent === ${name});
        if (index < 0) return null;
        cards[index].setAttribute('data-windows-qa-project', 'true');
        return '[data-windows-qa-project="true"]';
      })()`));
      await click(selector);
    }
    await until(() => evaluate(`!!document.querySelector('[aria-label="Python, pip, or uv command"]')`));
    if (await evaluate(`document.querySelector('.workspace-terminal-toggle')?.getAttribute('aria-expanded') === 'false'`)) {
      await click('.workspace-terminal-toggle');
    }
    await until(() => evaluate(`document.querySelector('.workspace-terminal-toggle')?.getAttribute('aria-expanded') === 'true'`));
    receipt.checks.actual_project_opened_in_normal_ui = true;
    if (!await evaluate(`document.body.innerText.includes(${JSON.stringify(args['project-name'])})`)) throw new Error('The requested project name is absent from the actual workspace.');
    let boundProject;
    if (savedBinomialEnabled) {
      boundProject = await evaluate(`(${readOpenedLocalConsole.toString()})(${JSON.stringify(args['project-name'])}, ${JSON.stringify(originalReceipt?.original_execution.project_id ?? null)})`);
      if (args.mode === 'create' && boundProject.history.length !== 0) {
        throw new Error('The fresh UI project already contains an execution.');
      }
    }
    if (args.mode === 'create') {
      await fill('[aria-label="Python, pip, or uv command"]', command);
      if (await evaluate(`document.querySelector('[aria-label="Python, pip, or uv command"]').value`) !== command) {
        throw new Error('The actual UI input did not retain the exact original command.');
      }
      await click('[aria-label="Run command"]');
      await until(() => evaluate(`[...document.querySelectorAll('.workspace-terminal-output')].some(node => node.innerText.includes('WINDOWS_NATIVE_UI_THREE_MODELS_OK')) && !document.querySelector('[aria-label="Python, pip, or uv command"]').disabled`), 300000);
      // Require the marker in output, rather than merely in the echoed command.
      const completed = await evaluate(`(() => {
        const output = [...document.querySelectorAll('.workspace-terminal-output')].map(node => node.textContent).join(' ');
        return output.includes('WINDOWS_NATIVE_UI_THREE_MODELS_OK');
      })()`);
      if (!completed) throw new Error('The visible terminal did not contain a successful three-model result.');
      receipt.checks.normal_ui_run_ols_logit_poisson_oracles_and_saved_state = true;
    } else {
      const history = await until(() => evaluate(`(() => {
        const output = [...document.querySelectorAll('.workspace-terminal-output')].map(node => node.textContent).join(' ');
        return output.includes('WINDOWS_NATIVE_UI_THREE_MODELS_OK');
      })()`));
      if (!history) throw new Error('Cold native reopen did not display the saved successful result.');
      receipt.checks.cold_reopen_visible_history_without_refit = true;
    }
    await until(() => evaluate(`document.querySelectorAll('.model-output').length >= 3`));
    receipt.model_tags = await evaluate(`[...document.querySelectorAll('.model-tag')].map(node => node.textContent)`);
    if (!['OLS', 'LOGIT', 'POISSON'].every(name => receipt.model_tags.includes(name))) throw new Error('All three persisted model tables were not rendered.');
    receipt.checks.three_model_tables_rendered = true;
    if (regularizedEnabled) {
      const output = await until(() => evaluate(`(() => {
        const stdout = [...document.querySelectorAll('.workspace-terminal-output')].map(node => node.textContent).join(String.fromCharCode(10));
        return (${regularizedProofFromStdout.toString()})(stdout, ${JSON.stringify(REGULARIZED_MARKER)});
      })()`), 300000);
      await until(() => evaluate(`document.querySelectorAll('.data-output').length === ${savedBinomialEnabled ? 12 : 4}`));
      const tables = [];
      for (let index = 0; index < 4; index++) {
        const table = await until(() => evaluate(`(() => {
          const node = [...document.querySelectorAll('.data-output')][${index}];
          if (!node) return null;
          node.scrollIntoView({block: 'center'});
          const rect = node.getBoundingClientRect();
          const presentation = node.querySelector('.latex-preview:not([hidden])') || node.querySelector('table');
          if (!presentation) return null;
          const content = presentation.getBoundingClientRect();
          const rows = (${renderedTableRows.toString()})(presentation);
          return {label: node.querySelector('.output-label')?.textContent,
            text: presentation.textContent, rows, visible: rect.width > 2 && rect.height > 2
              && rect.bottom > 0 && rect.top < innerHeight && content.width > 2 && content.height > 2,
            rectangle: {x: rect.x, y: rect.y, width: rect.width, height: rect.height}};
        })()`));
        tables.push(table);
      }
      receipt.regularized = verifyRegularizedPresentation(output, tables);
      const entries = await evaluate(`document.querySelectorAll('.workspace-terminal-entry').length`);
      if (entries !== 1) throw new Error('Four-family cold readback must retain exactly one original user command.');
      receipt.checks[args.mode === 'create'
        ? 'normal_ui_run_four_regularized_methods_complete_state_and_tables'
        : 'cold_reopen_four_regularized_tables_without_refit'] = true;
      receipt.checks.four_regularized_tables_rendered = true;
      receipt.original_user_command_count = entries;
    }
    if (savedBinomialEnabled) {
      await until(() => evaluate(`(() => {
        const stdout = [...document.querySelectorAll('.workspace-terminal-output')].map(node => node.textContent).join(String.fromCharCode(10));
        return stdout.split(/\\r?\\n/).filter(line => line.startsWith(${JSON.stringify(SAVED_BINOMIAL_MARKER)})).length === 1;
      })()`), 300000);
      const readback = await evaluate(`(${readOpenedLocalConsole.toString()})(${JSON.stringify(args['project-name'])}, ${JSON.stringify(boundProject.project_id)})`);
      receipt.original_execution = verifyOriginalLocalExecution(readback, command, boundProject.project_id,
        originalReceipt?.original_execution);
      const proof = savedBinomialProofFromExecution(receipt.original_execution.record, regularizedEnabled);
      if (proof.oracle_sha256 !== receipt.saved_binomial_oracle_sha256) throw new Error('The executed oracle differs from the exact inspected source.');
      receipt.checks.same_project_complete_original_execution_readback = true;
      receipt.checks.visible_saved_binomial_marker_prefix = true;
      await until(() => evaluate(`document.querySelectorAll('.model-output').length === 3 && document.querySelectorAll('.data-output').length === ${regularizedEnabled ? 12 : 8}`));
      const tables = [];
      for (let index = 0; index < 8; index++) {
        tables.push(await until(() => evaluate(`(() => {
          const node = [...document.querySelectorAll('.data-output')][${index + (regularizedEnabled ? 4 : 0)}];
          if (!node) return null;
          node.scrollIntoView({block: 'center'});
          const rect = node.getBoundingClientRect();
          const presentation = node.querySelector('.latex-preview:not([hidden])') || node.querySelector('table');
          if (!presentation) return null;
          const content = presentation.getBoundingClientRect();
          return {label: node.querySelector('.output-label')?.textContent, text: presentation.textContent,
            rows: (${renderedTableRows.toString()})(presentation).filter(row => row[1] !== 'Term'),
            visible: rect.width > 2 && rect.height > 2 && rect.bottom > 0 && rect.top < innerHeight
              && content.width > 2 && content.height > 2};
        })()`)));
      }
      receipt.saved_binomial = verifySavedBinomialPresentation(proof, tables, args['source-sha']);
      if (await evaluate(`document.querySelectorAll('.workspace-terminal-entry').length`) !== 1) throw new Error('Saved-binomial cold reopening must preserve one original command.');
      const preview = await evaluate(`document.querySelector('.workspace-terminal-command pre[aria-label="Command"]')?.textContent`);
      if (preview !== (command.length > 512 ? command.slice(0, 511) + '…' : command)) {
        throw new Error('The visible original UI command preview differs from the submitted command.');
      }
      receipt.original_user_command_count = 1;
      receipt.checks[args.mode === 'create' ? 'normal_ui_run_saved_binomial_full_state_and_tables'
        : 'cold_reopen_saved_binomial_tables_without_refit'] = true;
      receipt.checks.eight_saved_binomial_tables_rendered = true;
    }
    receipt.viewport = await evaluate(`({width: innerWidth, height: innerHeight, bodyWidth: document.body.scrollWidth})`);
    if (receipt.viewport.width < 800 || receipt.viewport.height < 500 || receipt.viewport.bodyWidth > receipt.viewport.width + 2) throw new Error('Native viewport is unusable or overflows.');
    receipt.status = 'passed';
  } catch (error) {
    receipt.status = 'error'; receipt.error = error.message; process.exitCode = 1;
  } finally {
    if (socket) socket.close();
    for (const entry of pending.values()) { clearTimeout(entry.timer); entry.reject(new Error('Owned CDP closed.')); }
    await writeFile(args.output, JSON.stringify(receipt, null, 2) + '\n');
    console.log(JSON.stringify({ status: receipt.status, mode: receipt.mode, checks: Object.keys(receipt.checks).length }));
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
