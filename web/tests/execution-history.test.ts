import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import type { WorkspaceClient } from "../src/api.ts";
import type { ExecutionRecord } from "../src/types.ts";
import type { HistoryPage } from "../src/ExecutionHistory.tsx";
import type { ResultSharingRecord } from "../src/result-sharing.ts";
const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  url: "http://localhost/",
});
for (const name of [
  "window",
  "document",
  "navigator",
  "HTMLElement",
  "HTMLInputElement",
  "Event",
  "MouseEvent",
])
  Object.defineProperty(globalThis, name, {
    value: dom.window[name as keyof typeof dom.window],
    configurable: true,
  });
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
const React = await import("react");
const { createRoot } = await import("react-dom/client");
const require = createRequire(import.meta.url),
  urls = new Map<string, string>();
const sourceRoot = new URL("../src/", import.meta.url);
function moduleUrl(name: string): string {
  if (urls.has(name)) return urls.get(name)!;
  const file = new URL(
    `${name}${existsSync(fileURLToPath(new URL(`${name}.tsx`, sourceRoot))) ? ".tsx" : ".ts"}`,
    sourceRoot,
  );
  let output = ts.transpileModule(readFileSync(file, "utf8"), {
    compilerOptions: {
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  output = output.replace(
    /(from\s*|import\s*)(["'])([^"']+)\2/g,
    (_whole, prefix, _quote, target: string) =>
      prefix +
      JSON.stringify(
        target.startsWith(".")
          ? moduleUrl(target.replace(/^\.\//, "").replace(/\.tsx?$/, ""))
          : pathToFileURL(require.resolve(target)).href,
      ),
  );
  const url =
    "data:text/javascript;base64," + Buffer.from(output).toString("base64");
  urls.set(name, url);
  return url;
}
const { default: ExecutionHistory } = await import(
  moduleUrl("ExecutionHistory")
);
const { ApiError } = await import(moduleUrl("api"));
const record = (id: string): ExecutionRecord => ({
  id,
  code: `print('${id}')`,
  created_at: "2026-10-01T00:00:00+00:00",
  status: "ok",
  stdout: "saved output",
  outputs: [],
  error: null,
  variables: [],
  duration_ms: 1,
  session_generation: 1,
});
const page = (
  ids: string[],
  cursor: string | null = null,
  start = "first-page",
): HistoryPage => ({
  runs: ids.map((id) => ({
    id,
    created_at: "2026-10-01T00:00:00+00:00",
    state: "finished",
    status: "ok",
    code_preview: `print('${id}')`,
    code_preview_truncated: false,
    actor_email: "editor@example.com",
  })),
  next_cursor: cursor,
  page_cursor: start,
  snapshot_at: "2026-10-06T00:00:00+00:00",
  scanned: ids.length,
});
function setup(
  handler: (path: string, signal?: AbortSignal) => Promise<unknown>,
  desktop = false,
  sharingRecords?: ReadonlyMap<string, ResultSharingRecord>,
) {
  document.body.innerHTML = "<div id='app'></div>";
  const root = createRoot(document.getElementById("app")!);
  const calls: string[] = [],
    opened: ExecutionRecord[] = [],
    selected: string[] = [];
  const client: WorkspaceClient = {
    projectId: "history-project",
    teams: true,
    desktop,
    connect: async () => ({ token: "", version: "test", environment: "team" }),
    cancelPending() {},
    async request<T>(path, options = {}) {
      calls.push(path);
      return (await handler(path, options.signal ?? undefined)) as T;
    },
  };
  return {
    calls,
    opened,
    selected,
    async mount() {
      await React.act(async () =>
        root.render(
          React.createElement(ExecutionHistory, {
            client,
            history: [record("local-run")],
            sharingRecords,
            onSelect: (id: string) => selected.push(id),
            onOpen: (value: ExecutionRecord) => opened.push(value),
          }),
        ),
      );
    },
    async close() {
      await React.act(() => root.unmount());
    },
  };
}
const button = (label: string) =>
  [...document.querySelectorAll<HTMLButtonElement>("button")].find((value) =>
    value.textContent?.includes(label),
  )!;
async function click(label: string) {
  await React.act(() => button(label).click());
}
async function input(value: string, selector = 'input:not([type="date"])') {
  const field = document.querySelector<HTMLInputElement>(selector)!;
  await React.act(() => {
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )!.set!.call(field, value);
    field.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
async function search() {
  await React.act(() =>
    document
      .querySelector("form")!
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })),
  );
}

test("older pages open the original saved record without execution, and previous uses its snapshot cursor", async () => {
  const f = setup(async (path) =>
    path.startsWith("/runs/")
      ? record("old-run")
      : new URLSearchParams(path.split("?")[1]).get("cursor") === "next-page"
        ? page(["old-run"], null, "next-page")
        : page(["recent-run"], "next-page"),
  );
  await f.mount();
  assert.match(document.body.textContent!, /recent-run/);
  await click("Older runs");
  assert.match(document.body.textContent!, /old-run/);
  assert.doesNotMatch(document.body.textContent!, /recent-run/);
  await click("Previous page");
  assert.equal(
    new URLSearchParams(f.calls.at(-1)!.split("?")[1]).get("cursor"),
    "first-page",
  );
  await click("Older runs");
  await click("print('old-run')");
  assert.deepEqual(f.opened, [record("old-run")]);
  assert.equal(f.calls.filter((path) => path.startsWith("/runs/")).length, 1);
  assert(!f.calls.some((path) => path.includes("execute")));
  await f.close();
});

test("search encodes code and date filters and restarts at the first page", async () => {
  const f = setup(async () => page(["match"]));
  await f.mount();
  await input("oe.ols(🌍) & x=1");
  await input("2026-10-01", 'input[type="date"]');
  await search();
  const params = new URLSearchParams(f.calls.at(-1)!.split("?")[1]);
  assert.equal(params.get("query"), "oe.ols(🌍) & x=1");
  assert.equal(params.get("since"), "2026-10-01");
  assert.equal(params.get("cursor"), null);
  await f.close();
});

test("late pages from an aborted search cannot replace a newer result", async () => {
  let resolve!: (value: unknown) => void;
  const pending = new Promise((done) => (resolve = done));
  const f = setup(async (path) =>
    new URLSearchParams(path.split("?")[1]).get("query") === "new"
      ? page(["new-result"])
      : pending,
  );
  await f.mount();
  await input("new");
  await search();
  assert.match(document.body.textContent!, /new-result/);
  await React.act(() => resolve(page(["stale-result"])));
  assert.doesNotMatch(document.body.textContent!, /stale-result/);
  await f.close();
});

test("a failed older page preserves existing rows and retries without skipping its cursor", async () => {
  let failures = 0;
  const f = setup(async (path) => {
    if (new URLSearchParams(path.split("?")[1]).get("cursor") === "next") {
      if (failures++ === 0) throw Error("Connection lost. Retry.");
      return page(["old"], null, "next");
    }
    return page(["recent"], "next");
  });
  await f.mount();
  await click("Older runs");
  assert.match(
    document.querySelector('[role="alert"]')!.textContent!,
    /Connection lost/,
  );
  assert.match(document.body.textContent!, /recent/);
  await click("Older runs");
  assert.match(document.body.textContent!, /old/);
  assert(!document.querySelector('[role="alert"]'));
  await f.close();
});

test("revoked access clears the history page", async () => {
  let n = 0;
  const f = setup(async () => {
    if (n++) throw new ApiError("Project access removed", 404);
    return page(["private-run"], "next");
  });
  await f.mount();
  await click("Older runs");
  assert.doesNotMatch(document.body.textContent!, /private-run/);
  assert.match(document.body.textContent!, /access removed/);
  await f.close();
});

test("bounded empty pages can advance and pending runs cannot open", async () => {
  let n = 0;
  const f = setup(async () => {
    if (!n++) return page([], "next");
    const value = page(["pending"], null, "next");
    value.runs[0].state = "running";
    value.runs[0].status = null;
    return value;
  });
  await f.mount();
  assert.match(document.body.textContent!, /Continue to older runs/);
  assert(!button("Older runs").disabled);
  await click("Older runs");
  assert(button("print('pending')").disabled);
  await f.close();
});

test("desktop preserves local selection and explicitly browses shared history", async () => {
  const f = setup(async () => page(["shared-run"]), true);
  await f.mount();
  assert.equal(f.calls.length, 0);
  await click("print('local-run')");
  assert.deepEqual(f.selected, ["local-run"]);
  await click("Shared history");
  assert.equal(f.calls.length, 1);
  assert.match(document.body.textContent!, /shared-run/);
  await click("This device");
  assert.match(document.body.textContent!, /local-run/);
  await f.close();
});

test("desktop history retains delivery status and shared archive browsing together", async () => {
  const sharing = new Map<string, ResultSharingRecord>([["local-run", {
    id: "local-run", actor_uid: "editor", state: "failed", retryable: true,
    error: { code: "INVALID_RESULT", message: "Retry sharing", status: 422 },
  }]]);
  const f = setup(async () => page(["shared-run"]), true, sharing);
  await f.mount();
  assert.match(button("print('local-run')").textContent!, /Sharing failed/);
  await click("print('local-run')");
  assert.deepEqual(f.selected, ["local-run"]);
  assert.equal(f.calls.length, 0);
  await click("Shared history");
  assert.match(document.body.textContent!, /shared-run/);
  assert.doesNotMatch(document.body.textContent!, /Sharing failed/);
  await click("This device");
  assert.match(button("print('local-run')").textContent!, /Sharing failed/);
  assert(!f.calls.some((path) => path.includes("execute")));
  await f.close();
});

test("search counts Unicode code points and preserves an oversized draft without sending it", async () => {
  const f = setup(async () => page(["match"]));
  await f.mount();
  const field = document.querySelector<HTMLInputElement>('input:not([type="date"])')!;
  assert.equal(field.hasAttribute("maxlength"), false);
  await input("🌍".repeat(256)); await search();
  assert.equal(new URLSearchParams(f.calls.at(-1)!.split("?")[1]).get("query"), "🌍".repeat(256));
  const before = f.calls.length;
  await input("🌍".repeat(257)); await search();
  assert.equal(f.calls.length, before); assert.equal(field.value, "🌍".repeat(257));
  assert.match(document.querySelector('[role="alert"]')!.textContent!, /256 characters/);
  await f.close();
});
