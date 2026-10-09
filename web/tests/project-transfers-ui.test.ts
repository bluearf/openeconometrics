import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import {
  createProjectTransferActions,
  type ProjectTransferDependencies,
} from "../src/project-transfers.ts";
import type { DatasetProfile } from "../src/types.ts";
const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  url: "http://localhost/",
  pretendToBeVisual: true,
});
for (const name of [
  "window",
  "document",
  "navigator",
  "HTMLElement",
  "Event",
  "MouseEvent",
])
  Object.defineProperty(globalThis, name, {
    value: dom.window[name as keyof typeof dom.window],
    configurable: true,
  });
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
const React = await import("react"),
  { createRoot } = await import("react-dom/client");
const require = createRequire(import.meta.url);
let output = ts
  .transpileModule(
    readFileSync(
      new URL("../src/ProjectTransfers.tsx", import.meta.url),
      "utf8",
    ),
    {
      compilerOptions: {
        jsx: ts.JsxEmit.ReactJSX,
        module: ts.ModuleKind.ESNext,
        target: ts.ScriptTarget.ES2022,
      },
    },
  )
  .outputText.replace(/import\s+["'][^"']+\.css["'];?/g, "");
output = output.replace(
  /(from\s*)(["'])([^"']+)\2/g,
  (_whole, prefix, _quote, target: string) =>
    prefix +
    JSON.stringify(
      target.startsWith(".")
        ? new URL(`../src/${target.replace(/^\.\//, "")}.ts`, import.meta.url)
            .href
        : pathToFileURL(require.resolve(target)).href,
    ),
);
const { default: ProjectTransfers } = await import(
  "data:text/javascript;base64," + Buffer.from(output).toString("base64")
);
const id = "a".repeat(32),
  cloudId = "b".repeat(32),
  sha = "c".repeat(64);
const view = {
  request_id: id,
  name: "observations.csv",
  size_bytes: 8_000_000,
  sha256: sha,
  state: "waiting",
  acknowledged_bytes: 4_000_000,
};
const file = {
  id: cloudId,
  name: view.name,
  size_bytes: view.size_bytes,
  data_hash: sha,
  transfer: "chunked-v1",
};
const imported = {
  id: "33333333-3333-4333-8333-333333333333",
  cloud_id: cloudId,
  sha256: sha,
  data_hash: "d".repeat(64),
  name: view.name,
} as DatasetProfile;
function deferred<T>() {
  let resolve!: (x: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
async function harness(
  opts: {
    empty?: boolean;
    readOnly?: boolean;
    disabled?: boolean;
    invoke?: ProjectTransferDependencies["invoke"];
  } = {},
) {
  const calls: string[] = [],
    ready: DatasetProfile[] = [];
  const actions = createProjectTransferActions({
    assertAccess() {},
    async importReady() {
      return imported;
    },
    async invoke(action, request) {
      calls.push(`${action}:${request || ""}`);
      if (opts.invoke) return opts.invoke(action, request);
      return action === "list"
        ? { status: 200, body: { transfers: opts.empty ? [] : [view] } }
        : action === "cancel"
          ? { status: 200, body: { cancelled: true, request_id: id } }
          : { status: 200, body: { state: "ready", file } };
    },
  });
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  const render = (next = actions) =>
    React.act(async () => {
      root.render(
        React.createElement(ProjectTransfers, {
          actions: next,
          readOnly: !!opts.readOnly,
          disabled: !!opts.disabled,
          onReady(value: DatasetProfile) {
            ready.push(value);
          },
        }),
      );
    });
  await render();
  const button = (text: string) => {
    const found = [...host.querySelectorAll<HTMLButtonElement>("button")].find(
      (x) => x.textContent === text,
    );
    assert.ok(found, text);
    return found;
  };
  const click = (text: string) => React.act(async () => button(text).click());
  const close = async () => {
    await React.act(async () => root.unmount());
    host.remove();
    actions.dispose();
  };
  return { host, actions, calls, ready, button, click, render, close };
}
test("pending transfer lives in a compact section, with pinned name and byte progress; reconnect only lists", async () => {
  const h = await harness();
  try {
    assert.match(h.host.textContent!, /observations.csv/);
    assert.match(h.host.textContent!, /50%/);
    assert.equal(h.host.querySelector("progress")!.value, 4_000_000);
    assert.deepEqual(h.calls, ["list:"]);
    await React.act(async () => window.dispatchEvent(new Event("online")));
    assert.deepEqual(h.calls, ["list:", "list:"]);
    await h.click("Devam et");
    assert.deepEqual(h.ready, [imported]);
    assert.equal(h.host.querySelector("section"), null);
  } finally {
    await h.close();
  }
});
test("no pending transfer renders no permanent toolbar; viewers can see status without mutation buttons", async () => {
  const empty = await harness({ empty: true });
  try {
    assert.equal(empty.host.querySelector("section"), null);
  } finally {
    await empty.close();
  }
  const viewer = await harness({ readOnly: true });
  try {
    assert.match(viewer.host.textContent!, /observations.csv/);
    assert.equal(viewer.host.querySelector("button"), null);
    assert.deepEqual(viewer.calls, ["list:"]);
  } finally {
    await viewer.close();
  }
});
test("cancel stays available during active resume and deferred cancellation is visible until explicit cleanup", async () => {
  const pending = deferred<{ status: number; body: unknown }>();
  let cancels = 0;
  const h = await harness({
    invoke: async (action) =>
      action === "list"
        ? { status: 200, body: { transfers: [view] } }
        : action === "resume"
          ? pending.promise
          : ++cancels === 1
            ? { status: 202, body: { ...view, state: "cancel_pending" } }
            : { status: 200, body: { cancelled: true, request_id: id } },
  });
  try {
    await h.click("Devam et");
    assert.equal(h.button("Devam et").disabled, true);
    assert.equal(h.button("İptal").disabled, false);
    await h.click("İptal");
    assert.match(h.host.textContent!, /İptal bekliyor/);
    assert.equal(h.button("Devam et").disabled, true);
    await React.act(async () => {
      pending.resolve({ status: 200, body: { state: "ready", file } });
    });
    assert.deepEqual(h.ready, []);
    await h.click("İptali tamamla");
    assert.equal(h.host.querySelector("section"), null);
  } finally {
    pending.resolve({ status: 202, body: view });
    await h.close();
  }
});
test("other workspace work disables resume without hiding cancellation", async () => {
  const h = await harness({ disabled: true });
  try {
    assert.equal(h.button("Devam et").disabled, true);
    assert.equal(h.button("İptal").disabled, false);
    await h.click("Devam et");
    assert.deepEqual(h.calls, ["list:"]);
  } finally {
    await h.close();
  }
});
test("changing project controller ignores an old pending resume callback", async () => {
  const pending = deferred<{ status: number; body: unknown }>();
  const h = await harness({
    invoke: async (action) =>
      action === "list"
        ? { status: 200, body: { transfers: [view] } }
        : pending.promise,
  });
  const next = createProjectTransferActions({
    assertAccess() {},
    async importReady() {
      return imported;
    },
    async invoke() {
      return { status: 200, body: { transfers: [] } };
    },
  });
  try {
    await h.click("Devam et");
    await h.render(next);
    await React.act(async () => {
      pending.resolve({ status: 200, body: { state: "ready", file } });
    });
    assert.deepEqual(h.ready, []);
    assert.equal(h.host.querySelector("section"), null);
  } finally {
    await h.close();
    next.dispose();
  }
});

let downloadOutput = ts
  .transpileModule(
    readFileSync(
      new URL("../src/ProjectFileDownload.tsx", import.meta.url),
      "utf8",
    ),
    {
      compilerOptions: {
        jsx: ts.JsxEmit.ReactJSX,
        module: ts.ModuleKind.ESNext,
        target: ts.ScriptTarget.ES2022,
      },
    },
  )
  .outputText.replace(/import\s+["'][^"']+\.css["'];?/g, "");
downloadOutput = downloadOutput.replace(
  /(from\s*)(["'])([^"']+)\2/g,
  (_whole, prefix, _quote, target: string) =>
    prefix +
    JSON.stringify(
      target.startsWith(".")
        ? new URL(`../src/${target.replace(/^\.\//, "")}.ts`, import.meta.url)
            .href
        : pathToFileURL(require.resolve(target)).href,
    ),
);
const { default: ProjectFileDownload } = await import(
  "data:text/javascript;base64," +
    Buffer.from(downloadOutput).toString("base64")
);
test("active download control sends one cancellation for its file ID and sanitizes native failure text", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  const calls: string[] = [];
  try {
    await React.act(async () =>
      root.render(
        React.createElement(ProjectFileDownload, {
          file: { id: cloudId, name: view.name },
          cancel: async (id: string) => {
            calls.push(id);
            throw new Error("Bearer token /Users/private/path");
          },
        }),
      ),
    );
    assert.match(host.textContent!, /İndiriliyor/);
    assert.match(host.textContent!, /observations.csv/);
    await React.act(async () =>
      host.querySelector<HTMLButtonElement>("button")!.click(),
    );
    assert.deepEqual(calls, [cloudId]);
    assert.match(host.textContent!, /Could not cancel/);
    assert.ok(!host.textContent!.includes("private"));
  } finally {
    await React.act(async () => root.unmount());
    host.remove();
  }
});
