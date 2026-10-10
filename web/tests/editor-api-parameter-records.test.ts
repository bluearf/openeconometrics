import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";
import { decodeApiEntry, type ApiParameter, type StoredApiEntry } from "../src/editor-api.ts";

// The wire vocabulary is asserted independently of the browser implementation.
const RECORDS: ApiParameter[] = [
  { name: "alpha", kind: "keyword-only", default: "0.05" },
  { name: "missing", kind: "keyword-only", default: "'raise'" },
  { name: "data", kind: "positional-or-keyword" },
  { name: "data", kind: "keyword-only" },
  { name: "y", kind: "keyword-only" },
  { name: "weights", kind: "keyword-only", default: "None" },
  { name: "result", kind: "positional-or-keyword" },
  { name: "device", kind: "keyword-only", default: "'cpu'" },
  { name: "x", kind: "keyword-only" },
  { name: "intercept", kind: "keyword-only", default: "True" },
  { name: "categorical", kind: "keyword-only", default: "None" },
  { name: "cluster", kind: "keyword-only", default: "None" },
  { name: "max_work", kind: "keyword-only", default: "100000000" },
  { name: "y", kind: "positional-or-keyword" },
  { name: "covariance", kind: "keyword-only", default: "None" },
  { name: "max_work", kind: "keyword-only", default: "50000000" },
  { name: "options", kind: "var-keyword" },
  { name: "time", kind: "keyword-only", default: "None" },
  { name: "level", kind: "keyword-only", default: "0.95" },
  { name: "x", kind: "keyword-only", default: "None" },
];
const ENTRY = { n: "~example", S: "()", d: "Complete Türkçe help." };

test("complete parameter record tokens restore every field and independent mutable objects", () => {
  const stored = { ...ENTRY, p: RECORDS.map((_, i) => i) };
  const snapshot = structuredClone(stored);
  assert.deepEqual(decodeApiEntry(stored).parameters, RECORDS);
  const changed = decodeApiEntry(stored);
  changed.parameters[0].default = "changed";
  changed.parameters[0].description = "changed";
  assert.deepEqual(decodeApiEntry(stored).parameters, RECORDS);
  const repeated = decodeApiEntry({ ...ENTRY, p: [0, 0, 5, 5] }).parameters;
  repeated[0].default = "changed";
  assert.equal(repeated[1].default, "0.05");
  assert.notEqual(repeated[2], repeated[3]);
  assert.deepEqual(stored, snapshot);
});

test("malformed active parameter records fail before destructuring and integer JSON numbers remain readable", () => {
  for (const token of [true, false, null, undefined, -1, 20, .5, -0, NaN, Infinity, -Infinity, "0", [], [[]]]) {
    assert.throws(() => decodeApiEntry({ ...ENTRY, p: [token] } as unknown as StoredApiEntry),
      /Compact parameter record needs a recognized integer token/);
  }
  assert.deepEqual(decodeApiEntry(JSON.parse('{"n":"~example","S":"()","d":"Complete help.","p":[0.0,19.0]}')).parameters,
    [RECORDS[0], RECORDS[19]]);
});

test("full and compact parameter lists retain precedence over malformed lower priority records", () => {
  const full: StoredApiEntry = { ...ENTRY, parameters: [0, 5], p: [true, null], P: false } as unknown as StoredApiEntry;
  const legacy: StoredApiEntry = { ...ENTRY, p: [0, 5], P: false } as unknown as StoredApiEntry;
  for (const stored of [full, legacy]) {
    const snapshot = structuredClone(stored);
    const decoded = decodeApiEntry(stored);
    assert.deepEqual(decoded.parameters, [RECORDS[0], RECORDS[5]]);
    assert.equal(Object.hasOwn(decoded, "p"), false);
    assert.equal(Object.hasOwn(decoded, "P"), false);
    assert.deepEqual(stored, snapshot);
  }
  assert.equal(decodeApiEntry({ ...ENTRY, P: 0 }).parameters[0].name, "json_data");
});

test("literal records retain annotations, choices, blanks, aliases and unrelated metadata", () => {
  const stored: StoredApiEntry = { ...ENTRY, p: [
    0,
    { name: "alpha", kind: "keyword-only", default: "0.05", annotation: "", description: "", choices: [] },
    { n: "extra", k: "k", v: false, a: "bool", d: "Türkçe.", c: ["False", "True"] },
    { name: "full", n: -1, kind: "positional-only", k: "k", default: "", V: -1, B: -1, D: -1,
      annotation: "Any", a: "ignored", description: "", d: "ignored", choices: [], c: ["ignored"] },
  ] };
  assert.deepEqual(decodeApiEntry(stored).parameters, [
    RECORDS[0],
    { name: "alpha", kind: "keyword-only", default: "0.05", annotation: "", description: "", choices: [] },
    { name: "extra", kind: "keyword-only", default: "False", annotation: "bool", description: "Türkçe.", choices: ["False", "True"] },
    { name: "full", kind: "positional-only", default: "", annotation: "Any", description: "", choices: [] },
  ]);
});

test("all 963 d92 catalog entries retain complete browser metadata after record token replacement", () => {
  const baseline = JSON.parse(execFileSync("git", ["show",
    "d92e12b4289e59439449e7653e24df11fd06343b:web/src/editor-api.json"],
  { cwd: new URL("../..", import.meta.url), encoding: "utf8", maxBuffer: 2_000_000 })) as StoredApiEntry[];
  const logical = baseline.map(decodeApiEntry);
  assert.equal(logical.length, 963);
  const exact = (parameter: ApiParameter, record: ApiParameter): boolean => {
    const keys = Object.keys(parameter);
    return keys.length === Object.keys(record).length
      && keys.every(key => parameter[key as keyof ApiParameter] === record[key as keyof ApiParameter]);
  };
  let count = 0;
  const compact = logical.map(entry => ({ ...entry, parameters: entry.parameters.map(parameter => {
    const token = RECORDS.findIndex(record => exact(parameter, record));
    if (token === -1) return parameter;
    count += 1;
    return token;
  }) }));
  assert.ok(count >= 2300);
  assert.deepEqual(compact.map(decodeApiEntry), logical);
});
