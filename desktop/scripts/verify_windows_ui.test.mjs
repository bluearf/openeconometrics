// Controller proof validation only; actual native Windows receipts come from Actions.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { REGULARIZED_MARKER, REGULARIZED_NAMES, regularizedCode, regularizedProofFromStdout,
  verifyRegularizedPresentation } from './verify_windows_ui.mjs';

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
