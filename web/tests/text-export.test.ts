import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { JSDOM } from "jsdom";
import { saveTextExport, MAX_TEXT_EXPORT_BYTES } from "../src/text-export.ts";
import { download } from "../src/api.ts";

function page(native = true) {
  const dom = new JSDOM("<!doctype html><html><body></body></html>", {
    url: "http://127.0.0.1:12345",
  });
  const w = dom.window;
  Object.assign(globalThis, {
    window: w,
    document: w.document,
    CustomEvent: w.CustomEvent,
  });
  Object.assign(w, { __TAURI_INTERNALS__: native ? {} : undefined });
  let blobCalls = 0,
    clicked = 0,
    revoked = 0;
  URL.createObjectURL = () => {
    blobCalls++;
    return "blob:fixture";
  };
  URL.revokeObjectURL = () => {
    revoked++;
  };
  w.HTMLAnchorElement.prototype.click = () => {
    clicked++;
  };
  w.setTimeout = (fn: () => void) => {
    fn();
    return 0;
  };
  return {
    w,
    dom,
    blobCalls: () => blobCalls,
    clicked: () => clicked,
    revoked: () => revoked,
  };
}
const text = "\\begin{tabular}{l}\nİstanbul α 𝛽\n\\end{tabular}\n";
function saved(content: string) {
  return {
    status: "saved",
    path: "/owned-fixture/results.tex",
    bytes: Buffer.byteLength(content),
    sha256: createHash("sha256").update(content).digest("hex"),
  };
}

test("native LaTeX/JSON/log use only the text broker and verify UTF-8 bytes/hash", async () => {
  const f = page();
  const args: unknown[] = [];
  Object.assign(f.w, {
    __TAURI__: {
      core: {
        invoke: async (
          command: string,
          a: { content: string; filename: string },
        ) => {
          assert.equal(command, "save_text_export");
          assert.deepEqual(Object.keys(a).sort(), ["content", "filename"]);
          args.push(a);
          return saved(a.content);
        },
      },
    },
  });
  for (const [content, name, type] of [
    [text, "results.tex", "application/x-tex"],
    [
      JSON.stringify({ label: "İstanbul", value: 42 }),
      "result.json",
      "application/json",
    ],
    ["# run\nUnicode α\n", "openecon-session.txt", "text/plain"],
  ]) {
    const r = await saveTextExport(content, name, type);
    assert.equal(r.status, "saved");
    assert.equal(r.bytes, Buffer.byteLength(content));
  }
  assert.equal(args.length, 3);
  assert.equal(f.blobCalls(), 0);
  assert.equal(f.clicked(), 0);
  f.dom.window.close();
});
test("cancel is a normal result with no saved-file claim or browser fallback", async () => {
  const f = page();
  Object.assign(f.w, {
    __TAURI__: {
      core: {
        invoke: async () => ({
          status: "cancelled",
          path: null,
          sha256: null,
          bytes: 0,
        }),
      },
    },
  });
  assert.equal((await saveTextExport(text, "results.tex")).status, "cancelled");
  assert.equal(f.clicked(), 0);
  f.dom.window.close();
});
test("unavailable/denied native broker never falls through to WebKit", async () => {
  const f = page();
  await assert.rejects(saveTextExport(text, "results.tex"), /desktop app/);
  Object.assign(f.w, {
    __TAURI__: {
      core: {
        invoke: async () => {
          throw "Only the local window can use this operation.";
        },
      },
    },
  });
  await assert.rejects(saveTextExport(text, "results.tex"));
  assert.equal(f.blobCalls(), 0);
  assert.equal(f.clicked(), 0);
  f.dom.window.close();
});
test("invalid name and byte limit are refused before invoking native code", async () => {
  const f = page();
  let calls = 0;
  Object.assign(f.w, {
    __TAURI__: {
      core: {
        invoke: async () => {
          calls++;
        },
      },
    },
  });
  for (const name of [
    "",
    ".",
    "..",
    "../results.tex",
    "a/b",
    "C:\\a",
    "a:b",
    "a\0b",
    "a\nb",
    "x.",
    "x ",
    "α".repeat(121),
  ])
    await assert.rejects(saveTextExport(text, name), /filename/);
  await assert.rejects(
    saveTextExport("α".repeat(MAX_TEXT_EXPORT_BYTES / 2 + 1), "results.tex"),
    /32 MiB/,
  );
  assert.equal(calls, 0);
  f.dom.window.close();
});
test("unverified native receipts and mismatched saved contents are errors", async () => {
  const f = page();
  for (const r of [
    null,
    {},
    { status: "cancelled", path: "saved", bytes: 0 },
    { ...saved(text), bytes: text.length },
    { ...saved(text), sha256: "0".repeat(64) },
  ]) {
    Object.assign(f.w, { __TAURI__: { core: { invoke: async () => r } } });
    await assert.rejects(saveTextExport(text, "results.tex"));
  }
  assert.equal(f.clicked(), 0);
  f.dom.window.close();
});
test("web browser retains a Blob download and cleans up; it does not assert saved", async () => {
  const f = page(false);
  const r = await saveTextExport(text, "results.tex", "application/x-tex");
  assert.deepEqual(r, { status: "browser" });
  assert.equal(f.blobCalls(), 1);
  assert.equal(f.clicked(), 1);
  assert.equal(f.revoked(), 1);
  assert.equal(f.w.document.querySelectorAll("a").length, 0);
  f.dom.window.close();
});
test("UI download reports errors without an unhandled rejected promise or false success", async () => {
  const f = page();
  const errors: string[] = [];
  f.w.addEventListener("openecon-export-error", (e: Event) =>
    errors.push((e as CustomEvent).detail),
  );
  Object.assign(f.w, {
    __TAURI__: {
      core: {
        invoke: async () => {
          throw "The selected folder could not be opened.";
        },
      },
    },
  });
  assert.equal((await download(text, "results.tex")).status, "failed");
  assert.deepEqual(errors, ["The selected folder could not be opened."]);
  assert.equal(f.blobCalls(), 0);
  f.dom.window.close();
});
