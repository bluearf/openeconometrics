import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  PROJECT_CREATE_LIMITS,
  validateProjectDraft,
} from "../src/project-create.ts";

const cases = JSON.parse(
  readFileSync(
    new URL("../../tests/fixtures/project-create.json", import.meta.url),
    "utf8",
  ),
);

test("creation uses the server's 100/500 Unicode code-point limits", () => {
  assert.deepEqual(PROJECT_CREATE_LIMITS, { name: 100, description: 500 });
  const draft = validateProjectDraft("🧪".repeat(100), "🌍".repeat(500));
  assert.deepEqual(draft.lengths, { name: 100, description: 500 });
  assert.deepEqual(draft.errors, {});
});

for (const fixture of cases)
  test(`shared frontend/API creation contract: ${fixture.id}`, () => {
    const draft = validateProjectDraft(fixture.name, fixture.description);
    if (fixture.valid) {
      assert.deepEqual(draft.errors, {});
      assert.deepEqual(draft.input, {
        name: fixture.normalized_name,
        description: fixture.normalized_description,
      });
    } else {
      assert.deepEqual(Object.keys(draft.errors).sort(), fixture.fields.sort());
      assert.ok(
        Object.values(draft.errors).every(
          (error) => typeof error === "string" && error.length > 0,
        ),
      );
    }
  });
