/** Future browser/Node metadata tests; not executed in source-only review. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { decodeEditorWire } from "../src/editor-catalog-intern.ts";

const fixtures = JSON.parse(readFileSync(new URL("../../tests/fixtures/editor-catalog-intern-v2-pair-order.json", import.meta.url), "utf8"));

test("Python escaped pairs admit integer-like prototype Unicode and astral keys without field loss", () => {
  for (const item of fixtures.positive) {
    const decoded = decodeEditorWire(item.raw);
    assert.deepEqual(JSON.parse(JSON.stringify(decoded)), item.logical);
    if (item.name === "nested_tags_proto_and_ordered_parameter_arrays") {
      const object = decoded[0] as Record<string, unknown>;
      assert.equal(Object.getPrototypeOf(object), null);
      assert.deepEqual(object.ordered_parameters, [{ name: "second" }, { name: "first" }].map(value => Object.assign(Object.create(null), value)));
      assert.equal(({} as Record<string, unknown>).polluted, undefined);
    }
  }
  for (const item of fixtures.negative) assert.throws(() => decodeEditorWire(item.raw));
});

test("all original Python raw-wire negatives also refuse in browser strict-text admission", () => {
  const source = new URL("../../tests/fixtures/editor-catalog-intern-original-v1-negative-fixtures.json", import.meta.url);
  for (const item of JSON.parse(readFileSync(source, "utf8"))) assert.throws(() => decodeEditorWire(item.raw));
});
