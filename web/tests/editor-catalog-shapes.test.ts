/** Future Node/browser-source controls; not executed by this proposal. */
import assert from "node:assert/strict";
import { isDeepStrictEqual } from "node:util";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { decodeEditorWire, encodeShapedEditorWire } from "../src/editor-catalog-shapes.ts";

import rawEditorWire from "../src/editor-api.wire.ts";
import { API_CATALOG, decodeApiEntry, type StoredApiEntry } from "../src/editor-api.ts";

// Current wire budget (raised from 500,000 bytes on 10 Oct 2026 by owner decision).
// Historical binding fixtures keep their recorded caps for their own baselines.
const CURRENT_WIRE_CAP_BYTES = 2_000_000;

// Undo approved method wave 2 so each earlier binding below still compares
// exactly against its own baseline.
const wave2Bindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-methods-wave2-approved-changes.json", import.meta.url), "utf8"));
const wave2Added = new Set<string>(wave2Bindings.added_names);
function beforeWave2(entries: unknown[]): unknown[] {
  entries = beforeNgPerronHelp(entries);
  return entries.filter(entry => !wave2Added.has(decodeApiEntry(entry as StoredApiEntry).name)).map(entry => {
    const changed = wave2Bindings.changed_entries[decodeApiEntry(entry as StoredApiEntry).name];
    if (!changed) return entry;
    assert.deepEqual(JSON.parse(JSON.stringify(entry)), changed.after_compact);
    assert.deepEqual(decodeApiEntry(entry as StoredApiEntry), changed.after_logical);
    return changed.before_compact;
  });
}
function beforeWave2Logical<T extends { name: string }>(entries: T[]): T[] {
  entries = entries.map(beforeNgPerronLogical);
  return entries.filter(entry => !wave2Added.has(entry.name)).map(entry => wave2Bindings.changed_entries[entry.name]?.before_logical ?? entry);
}

// Undo approved method wave 1 so each earlier binding below still compares
// exactly against its own baseline.
const wave1Bindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-methods-wave1-approved-changes.json", import.meta.url), "utf8"));
const wave1Added = new Set<string>(wave1Bindings.added_names);
function beforeWave1(entries: unknown[]): unknown[] {
  return entries.filter(entry => !wave1Added.has(decodeApiEntry(entry as StoredApiEntry).name)).map(entry => {
    const changed = wave1Bindings.changed_entries[decodeApiEntry(entry as StoredApiEntry).name];
    if (!changed) return entry;
    assert.deepEqual(JSON.parse(JSON.stringify(entry)), changed.after_compact);
    assert.deepEqual(decodeApiEntry(entry as StoredApiEntry), changed.after_logical);
    return changed.before_compact;
  });
}
function beforeWave1Logical<T extends { name: string }>(entries: T[]): T[] {
  return entries.filter(entry => !wave1Added.has(entry.name)).map(entry => wave1Bindings.changed_entries[entry.name]?.before_logical ?? entry);
}

const weibullBindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-main2f2548-Weibull1415-bindings.json", import.meta.url), "utf8"));
const historicalWeibullBindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-mainc41b4f-Weibull1417-bindings.json", import.meta.url), "utf8"));
const currentWeibullBindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-main369d84-Weibull1420-bindings.json", import.meta.url), "utf8"));
const weibullNames = new Set<string>(weibullBindings.complete_added_Weibull_names_in_order);
const bootstrapNames = new Set(["openecon.conjoint_bootstrap_fit", "openecon.conjoint_bootstrap_importance", "openecon.conjoint_bootstrap_shares"]);
const ngperronBindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-ngperron136-approved-changes.json", import.meta.url), "utf8"));
function beforeNgPerronHelp(entries: unknown[]): unknown[] {
  return entries.map(entry => {
    const changed = ngperronBindings.changed_entries[decodeApiEntry(entry as StoredApiEntry).name];
    if (!changed) return entry;
    // Nested historical filters may pass an entry that is already restored.
    if (isDeepStrictEqual(JSON.parse(JSON.stringify(entry)), changed.before_compact)) return entry;
    assert.deepEqual(JSON.parse(JSON.stringify(entry)), changed.after_compact);
    assert.deepEqual(decodeApiEntry(entry as StoredApiEntry), changed.after_logical);
    return changed.before_compact;
  });
}
function beforeNgPerronLogical<T extends { name: string }>(entry: T): T {
  return ngperronBindings.changed_entries[entry.name]?.before_logical ?? entry;
}
function beforeBootstrap(entries: unknown[]): unknown[] {
  return entries.filter(entry => !bootstrapNames.has(decodeApiEntry(entry as StoredApiEntry).name));
}
function withoutWeibull(entries: unknown[]): unknown[] {
  return beforeBootstrap(entries).filter(entry => !weibullNames.has(decodeApiEntry(entry as StoredApiEntry).name));
}

// Reconstruct only the two explicitly changed conjoint help entries for historical
// whole-object preservation; a separate test below checks every current object.
const conjointBindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-conjoint1392-approved-changes.json", import.meta.url), "utf8"));
const contrastAddedNames = ["openecon.canon_bootstrap_score_contrasts", "openecon.factor_bootstrap_score_contrasts", "openecon.pca_fweight_bootstrap_score_contrasts"];
const contrastAdded = new Set(contrastAddedNames);
function beforeContrasts(entries: unknown[]): unknown[] {
  return beforeBootstrap(entries).filter(entry => !contrastAdded.has(decodeApiEntry(entry as StoredApiEntry).name));
}
const conjointAdded = new Set<string>(conjointBindings.added_names);
function beforeConjoint(entries: unknown[]): unknown[] {
  return beforeContrasts(entries).filter(entry => !conjointAdded.has(decodeApiEntry(entry as StoredApiEntry).name)).map(entry => {
    const name = decodeApiEntry(entry as StoredApiEntry).name;
    const changed = conjointBindings.changed_entries[name];
    if (!changed) return entry;
    assert.deepEqual(JSON.parse(JSON.stringify(entry)), changed.after_compact);
    assert.deepEqual(decodeApiEntry(entry as StoredApiEntry), changed.after_logical);
    return changed.before_compact;
  });
}

const fixtures = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-shapes-v1.json", import.meta.url), "utf8"));
test("whole reserved, integer-like, prototype-looking and astral keys plus array order survive", () => {
  for (const item of fixtures.positive) {
    assert.deepEqual(JSON.parse(JSON.stringify(decodeEditorWire(item.raw))), item.logical);
    assert.equal(encodeShapedEditorWire(item.logical), item.raw);
    const restored = decodeEditorWire(item.raw);
    for (const object of restored) if (object !== null && typeof object === "object" && !Array.isArray(object)) {
      assert.equal(Object.getPrototypeOf(object), null);
    }
  }
  assert.equal(Object.hasOwn(Object.prototype, "polluted"), false);
});
test("every malformed and noncanonical shape wire is refused including raw integer lexemes", () => {
  for (const item of fixtures.negative) assert.throws(() => decodeEditorWire(item.raw), item.id);
});
test("old raw arrays and repaired intern-v1 routes and all original negatives remain exact", () => {
  for (const item of fixtures.legacy_arrays) assert.deepEqual(JSON.parse(JSON.stringify(decodeEditorWire(item.raw))), item.logical);
  const prior = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-intern-v2-pair-order.json", import.meta.url), "utf8"));
  for (const item of prior.positive) assert.deepEqual(JSON.parse(JSON.stringify(decodeEditorWire(item.raw))), item.logical);
  for (const item of prior.negative) assert.throws(() => decodeEditorWire(item.raw), item.id);
  const negatives = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-intern-original-v1-negative-fixtures.json", import.meta.url), "utf8"));
  for (const item of negatives) assert.throws(() => decodeEditorWire(item.raw), item.id);
});
test("prototype/accessor and original wire/decoded/resource guards remain active", () => {
  assert.throws(() => encodeShapedEditorWire([Object.create({ polluted: true })]));
  assert.throws(() => encodeShapedEditorWire([Object.defineProperty({}, "data", { get() { throw new Error("getter executed"); } })]));
  assert.throws(() => encodeShapedEditorWire([{ large: "x".repeat(2_000_000) }]));
  for (const value of [NaN, Infinity, -0, Number.MAX_SAFE_INTEGER+1]) assert.throws(() => encodeShapedEditorWire([{ value }]));
});
test("complete historical1329 saved wire decodes to every original compact field", () => {
  const bindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-shapes-full-bindings.json", import.meta.url), "utf8"));
  const raw = readFileSync(new URL("../../" + bindings.whole1329_shaped_wire, import.meta.url), "utf8");
  const compact = JSON.parse(readFileSync(new URL("../../" + bindings.whole1329_legacy_wire, import.meta.url), "utf8"));
  assert.equal(new TextEncoder().encode(raw).length, 471027);
  assert.deepEqual(JSON.parse(JSON.stringify(decodeEditorWire(raw))), compact);
  assert.equal(encodeShapedEditorWire(compact), raw);
});

test("current6f2 whole1251 objects survive the complete BVAR1365 subset of the browser catalog", () => {
  const bmaBindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-main4634-BMA1390-bindings.json", import.meta.url), "utf8"));
  const bmaNames = new Set<string>(bmaBindings.complete_added_BMA_canonical_names_in_order);
  const bindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-current6f2-BVAR1365-bindings.json", import.meta.url), "utf8"));
  const baselineRaw = readFileSync(new URL("../../" + bindings.original_baseline, import.meta.url), "utf8");
  const raw = readFileSync(new URL("../src/editor-api.json", import.meta.url), "utf8");
  assert.equal(rawEditorWire, raw);
  assert.equal(new TextEncoder().encode(baselineRaw).length, bindings.original_baseline_bytes);
  assert.equal(new TextEncoder().encode(raw).length < CURRENT_WIRE_CAP_BYTES, true);
  const baseline = decodeEditorWire(baselineRaw);
  const current = beforeConjoint(withoutWeibull(beforeWave1(beforeWave2(decodeEditorWire(rawEditorWire))))).filter(entry => !bmaNames.has(decodeApiEntry(entry as StoredApiEntry).name));
  const name = (entry: unknown) => decodeApiEntry(entry as StoredApiEntry).name;
  const names = new Set(baseline.map(name));
  assert.equal(baseline.length, 1251);
  assert.equal(names.size, baseline.length);
  assert.equal(current.length, 1365);
  assert.equal(new Set(current.map(name)).size, current.length);
  // Compare every original compact field as well as all expanded help fields.
  const retained = current.filter(entry => names.has(name(entry)));
  assert.deepEqual(JSON.parse(JSON.stringify(retained)), JSON.parse(JSON.stringify(baseline)));
  assert.deepEqual(retained.map(entry => decodeApiEntry(entry as StoredApiEntry)),
    baseline.map(entry => decodeApiEntry(entry as StoredApiEntry)));
  const added = current.filter(entry => !names.has(name(entry)));
  assert.equal(added.length, 114);
  assert.deepEqual(added.map(name), bindings.complete_added_BVAR_canonical_names_in_order);
  assert.deepEqual(beforeConjoint(withoutWeibull(beforeWave1(beforeWave2(decodeEditorWire(rawEditorWire))))).map(entry => decodeApiEntry(entry as StoredApiEntry)).filter(entry => !bmaNames.has(entry.name)), current.map(entry => decodeApiEntry(entry as StoredApiEntry)));
});


test("whole main4634 BVAR1365 objects and exact BMA25 additions survive the current1390 catalog", () => {
  const bindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-main4634-BMA1390-bindings.json", import.meta.url), "utf8"));
  const baselineRaw = readFileSync(new URL("../../" + bindings.original_baseline, import.meta.url), "utf8");
  const actualRaw = readFileSync(new URL("../src/editor-api.json", import.meta.url), "utf8");
  assert.equal(rawEditorWire, actualRaw);
  assert.equal(new TextEncoder().encode(baselineRaw).length, bindings.original_baseline_bytes);
  assert.equal(new TextEncoder().encode(actualRaw).length <= CURRENT_WIRE_CAP_BYTES, true);
  const baseline = decodeEditorWire(baselineRaw);
  const current = beforeConjoint(withoutWeibull(beforeWave1(beforeWave2(decodeEditorWire(actualRaw)))));
  const name = (entry: unknown) => decodeApiEntry(entry as StoredApiEntry).name;
  const names = new Set(baseline.map(name));
  assert.equal(baseline.length, bindings.original_baseline_objects);
  assert.equal(baseline.length, 1365);
  assert.equal(names.size, baseline.length);
  assert.equal(current.length, bindings.current_objects);
  assert.equal(current.length, 1390);
  assert.equal(new Set(current.map(name)).size, current.length);
  const retained = current.filter(entry => names.has(name(entry)));
  assert.deepEqual(JSON.parse(JSON.stringify(retained)), JSON.parse(JSON.stringify(baseline)));
  assert.deepEqual(retained.map(entry => decodeApiEntry(entry as StoredApiEntry)),
    baseline.map(entry => decodeApiEntry(entry as StoredApiEntry)));
  const added = current.filter(entry => !names.has(name(entry)));
  assert.equal(added.length, 25);
  assert.deepEqual(added.map(name), bindings.complete_added_BMA_canonical_names_in_order);
  assert.deepEqual(beforeConjoint(withoutWeibull(beforeWave1(beforeWave2(decodeEditorWire(rawEditorWire))))).map(entry => decodeApiEntry(entry as StoredApiEntry)), current.map(entry => decodeApiEntry(entry as StoredApiEntry)));
});


test("current conjoint1392 retains every other object and both complete added public targets", () => {
  const complete = beforeWave1(beforeWave2(decodeEditorWire(rawEditorWire)));
  const actual = beforeContrasts(withoutWeibull(complete));
  assert.equal(actual.length, conjointBindings.current_count);
  assert.equal(beforeConjoint(actual).length, conjointBindings.before_count);
  assert.deepEqual(actual.filter(entry => conjointAdded.has(decodeApiEntry(entry as StoredApiEntry).name))
    .map(entry => decodeApiEntry(entry as StoredApiEntry).name), conjointBindings.added_names);
  assert.deepEqual(beforeWave1Logical(beforeWave2Logical(Object.values(API_CATALOG))), complete.map(entry => decodeApiEntry(entry as StoredApiEntry)));
});


test("whole current main BMA1390 and all25 Weibull entries preserve complete objects and source order", () => {
  const baselineRaw = readFileSync(new URL("../../" + weibullBindings.original_baseline, import.meta.url), "utf8");
  const actualRaw = readFileSync(new URL("../src/editor-api.json", import.meta.url), "utf8");
  assert.equal(rawEditorWire, actualRaw);
  assert.equal(new TextEncoder().encode(baselineRaw).length, weibullBindings.original_baseline_bytes);
  assert.equal(new TextEncoder().encode(actualRaw).length <= CURRENT_WIRE_CAP_BYTES, true);
  const baseline = decodeEditorWire(baselineRaw);
  const current = beforeConjoint(beforeWave1(beforeWave2(decodeEditorWire(actualRaw))));
  const name = (entry: unknown) => decodeApiEntry(entry as StoredApiEntry).name;
  const names = new Set(baseline.map(name));
  assert.equal(baseline.length, 1390);
  assert.equal(current.length, 1415);
  assert.equal(new Set(current.map(name)).size, current.length);
  const retained = current.filter(entry => names.has(name(entry)));
  assert.deepEqual(JSON.parse(JSON.stringify(retained)), JSON.parse(JSON.stringify(baseline)));
  assert.deepEqual(retained.map(entry => decodeApiEntry(entry as StoredApiEntry)), baseline.map(entry => decodeApiEntry(entry as StoredApiEntry)));
  const added = current.filter(entry => !names.has(name(entry)));
  assert.equal(added.length, 25);
  assert.deepEqual(added.map(name), weibullBindings.complete_added_Weibull_names_in_order);
  assert.deepEqual(beforeConjoint(beforeWave1(beforeWave2(decodeEditorWire(rawEditorWire)))).map(entry => decodeApiEntry(entry as StoredApiEntry)), current.map(entry => decodeApiEntry(entry as StoredApiEntry)));
});


test("whole current conjoint main1392 and all25 Weibull entries preserve complete objects and source order", () => {
  const baselineRaw = readFileSync(new URL("../../" + historicalWeibullBindings.original_baseline, import.meta.url), "utf8");
  const actualRaw = readFileSync(new URL("../src/editor-api.json", import.meta.url), "utf8");
  assert.equal(rawEditorWire, actualRaw);
  assert.equal(new TextEncoder().encode(baselineRaw).length, historicalWeibullBindings.original_baseline_bytes);
  assert.equal(new TextEncoder().encode(actualRaw).length <= CURRENT_WIRE_CAP_BYTES, true);
  const baseline = decodeEditorWire(baselineRaw);
  const current = beforeContrasts(beforeWave1(beforeWave2(decodeEditorWire(actualRaw))));
  const name = (entry: unknown) => decodeApiEntry(entry as StoredApiEntry).name;
  const names = new Set(baseline.map(name));
  assert.equal(baseline.length, 1392);
  assert.equal(current.length, 1417);
  assert.equal(new Set(current.map(name)).size, current.length);
  const retained = current.filter(entry => names.has(name(entry)));
  assert.deepEqual(JSON.parse(JSON.stringify(retained)), JSON.parse(JSON.stringify(baseline)));
  assert.deepEqual(retained.map(entry => decodeApiEntry(entry as StoredApiEntry)), baseline.map(entry => decodeApiEntry(entry as StoredApiEntry)));
  const added = current.filter(entry => !names.has(name(entry)));
  assert.equal(added.length, 25);
  assert.deepEqual(added.map(name), historicalWeibullBindings.complete_added_Weibull_names_in_order);
  assert.deepEqual(beforeContrasts(beforeWave1(beforeWave2(decodeEditorWire(rawEditorWire)))).map(entry => decodeApiEntry(entry as StoredApiEntry)), current.map(entry => decodeApiEntry(entry as StoredApiEntry)));
});


test("whole current main1395 and all25 Weibull entries preserve complete objects and source order", () => {
  const baselineRaw = readFileSync(new URL("../../" + currentWeibullBindings.original_baseline, import.meta.url), "utf8");
  const actualRaw = readFileSync(new URL("../src/editor-api.json", import.meta.url), "utf8");
  assert.equal(rawEditorWire, actualRaw);
  assert.equal(new TextEncoder().encode(baselineRaw).length, currentWeibullBindings.original_baseline_bytes);
  assert.equal(new TextEncoder().encode(actualRaw).length <= CURRENT_WIRE_CAP_BYTES, true);
  const baseline = decodeEditorWire(baselineRaw);
  const current = beforeBootstrap(beforeWave1(beforeWave2(decodeEditorWire(actualRaw))));
  const name = (entry: unknown) => decodeApiEntry(entry as StoredApiEntry).name;
  const names = new Set(baseline.map(name));
  assert.equal(baseline.length, 1395);
  assert.equal(current.length, 1420);
  assert.equal(new Set(current.map(name)).size, current.length);
  const retained = current.filter(entry => names.has(name(entry)));
  assert.deepEqual(JSON.parse(JSON.stringify(retained)), JSON.parse(JSON.stringify(baseline)));
  assert.deepEqual(retained.map(entry => decodeApiEntry(entry as StoredApiEntry)), baseline.map(entry => decodeApiEntry(entry as StoredApiEntry)));
  const added = current.filter(entry => !names.has(name(entry)));
  assert.equal(added.length, 25);
  assert.deepEqual(added.map(name), currentWeibullBindings.complete_added_Weibull_names_in_order);
  assert.deepEqual(beforeWave1Logical(beforeWave2Logical(Object.values(API_CATALOG))).filter(entry => !bootstrapNames.has(entry.name)), current.map(entry => decodeApiEntry(entry as StoredApiEntry)));
});


test("three simultaneous score helpers preserve all1392 original whole objects and the current browser catalog", () => {
  const bindings = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-score-contrasts1395-bindings.json", import.meta.url), "utf8"));
  const actual = withoutWeibull(beforeWave1(beforeWave2(decodeEditorWire(rawEditorWire))));
  const prior = beforeContrasts(actual);
  assert.equal(prior.length, bindings.original_objects);
  assert.equal(actual.length, bindings.current_objects);
  assert.equal(createHash("sha256").update(JSON.stringify(prior)).digest("hex"), bindings.original_compact_objects_sha256);
  assert.deepEqual(actual.filter(entry => contrastAdded.has(decodeApiEntry(entry as StoredApiEntry).name))
    .map(entry => decodeApiEntry(entry as StoredApiEntry).name), contrastAddedNames);
  assert.deepEqual(beforeWave1Logical(beforeWave2Logical(Object.values(API_CATALOG))).filter(entry => !weibullNames.has(entry.name) && !bootstrapNames.has(entry.name)), actual.map(entry => decodeApiEntry(entry as StoredApiEntry)));
});

test("all1420 main objects survive three complete conjoint bootstrap additions", () => {
  const plan = JSON.parse(readFileSync(new URL("../../docs/econometrics/merge-gate-conjoint-bootstrap160-2026-10-10.json", import.meta.url), "utf8"));
  const baselineRaw = execFileSync("git", ["show", `${plan.source_parent_commit}:web/src/editor-api.json`], { encoding: "utf8" });
  const baseline = decodeEditorWire(baselineRaw);
  const current = beforeWave1(beforeWave2(decodeEditorWire(rawEditorWire)));
  assert.equal(baseline.length, 1420);
  assert.equal(current.length, 1423);
  assert.deepEqual(JSON.parse(JSON.stringify(beforeBootstrap(current))), JSON.parse(JSON.stringify(baseline)));
  assert.deepEqual(current.filter(entry => bootstrapNames.has(decodeApiEntry(entry as StoredApiEntry).name)).map(entry => decodeApiEntry(entry as StoredApiEntry).name), [...bootstrapNames]);
  assert.deepEqual(beforeWave1Logical(beforeWave2Logical(Object.values(API_CATALOG))), current.map(entry => decodeApiEntry(entry as StoredApiEntry)));
});


test("method wave 1 adds 38 targets, changes 3 help entries and retains every main1423 object", () => {
  const actual = beforeWave2(decodeEditorWire(rawEditorWire));
  const prior = beforeWave1(actual);
  assert.equal(actual.length, wave1Bindings.current_count);
  assert.equal(prior.length, wave1Bindings.before_count);
  assert.equal(createHash("sha256").update(JSON.stringify(prior)).digest("hex"), wave1Bindings.before_compact_objects_sha256);
  assert.deepEqual(actual.map(entry => decodeApiEntry(entry as StoredApiEntry).name).filter(name => wave1Added.has(name)), wave1Bindings.added_names);
  assert.deepEqual(Object.keys(wave1Bindings.changed_entries).sort(), ["openecon.arch", "openecon.asymcausality", "openecon.bai_perron"]);
  assert.equal(new TextEncoder().encode(rawEditorWire).length <= CURRENT_WIRE_CAP_BYTES, true);
  assert.deepEqual(beforeWave2Logical(Object.values(API_CATALOG)), actual.map(entry => decodeApiEntry(entry as StoredApiEntry)));
});


test("method wave 2 adds 20 targets, changes 0 help entries and retains every main1461 object", () => {
  const actual = decodeEditorWire(rawEditorWire);
  const prior = beforeWave2(actual);
  assert.equal(actual.length, wave2Bindings.current_count);
  assert.equal(prior.length, wave2Bindings.before_count);
  assert.equal(createHash("sha256").update(JSON.stringify(prior)).digest("hex"), wave2Bindings.before_compact_objects_sha256);
  assert.deepEqual(actual.map(entry => decodeApiEntry(entry as StoredApiEntry).name).filter(name => wave2Added.has(name)), wave2Bindings.added_names);
  assert.deepEqual(Object.keys(wave2Bindings.changed_entries).sort(), []);
  assert.equal(new TextEncoder().encode(rawEditorWire).length <= CURRENT_WIRE_CAP_BYTES, true);
  assert.deepEqual(Object.values(API_CATALOG), actual.map(entry => decodeApiEntry(entry as StoredApiEntry)));
});
