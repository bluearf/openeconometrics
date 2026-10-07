import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
import type { WorkspaceClient } from "../src/api.ts";
import { ApiError } from "../src/api.ts";
import type { ResultSharingSnapshot } from "../src/result-sharing.ts";

const dom = new JSDOM("<html><body></body></html>", { url: "http://localhost/" });
for (const name of ["window", "document", "navigator", "HTMLElement", "Event", "MouseEvent"])
  Object.defineProperty(globalThis, name, { value: dom.window[name as keyof typeof dom.window], configurable: true });
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
const React = await import("react");
const { createRoot } = await import("react-dom/client");
const require = createRequire(import.meta.url);
const source = ts.transpileModule(readFileSync(new URL("../src/ResultSharingPanel.tsx", import.meta.url), "utf8"), {
  compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText.replace(/from (["'])([^"']+)\1/g, (_match, _quote, name: string) =>
  `from ${JSON.stringify(name.startsWith(".")
    ? new URL(`../src/${name.replace(/^\.\//, "")}.ts`, import.meta.url).href
    : pathToFileURL(require.resolve(name)).href)}`);
const { default: Panel } = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
function snapshot(state: "pending" | "shared" | "failed" | "local_only", status?: number): ResultSharingSnapshot {
  return { pending_count: state === "pending" ? 1 : 0, failed_count: state === "failed" ? 1 : 0,
    records: [{ id: "run-1", actor_uid: "owner", state, retryable: ["pending", "failed"].includes(state),
      ...(status ? { error: { status, code: "TEST_ERROR", message: "Saved here; sharing needs attention." } } : {}) }] };
}
function setup(initial = snapshot("failed"), options: { readOnly?: boolean; onRetry?: () => Promise<unknown>;
  retrySnapshot?: ResultSharingSnapshot;
  onRead?: (count: number, value: ResultSharingSnapshot) => Promise<ResultSharingSnapshot> } = {}) {
  document.body.innerHTML = "<div id='app'></div>";
  const listeners = new Set<(value: ResultSharingSnapshot) => void>();
  const calls: string[] = [];
  let value = initial;
  let denied = 0, history = 0, reads = 0;
  const client: WorkspaceClient = {
    projectId: "a".repeat(32), teams: true, desktop: true,
    connect: async () => ({ token: "", version: "test", environment: "local" }), cancelPending() {},
    subscribeResultSharing(listener) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    async request<T>(path) {
      calls.push(path);
      if (path.endsWith("/retry")) {
        await options.onRetry?.();
        value = options.retrySnapshot ?? snapshot("shared");
        for (const listener of listeners) listener(value);
        return {} as T;
      }
      return (options.onRead ? await options.onRead(++reads, value) : value) as T;
    },
  };
  const root = createRoot(document.querySelector("#app")!);
  const props = { client, resultId: "run-1", readOnly: options.readOnly ?? false, ready: true,
    onOpenHistory: () => { history++; }, onSnapshot() {}, onAccessDenied: () => { denied++; } };
  return { root, client, calls, listeners, props,
    get denied() { return denied; }, get history() { return history; },
    async render(next = props) { await React.act(async () => { root.render(React.createElement(Panel, next)); }); },
    async emit(next: ResultSharingSnapshot) { value = next; await React.act(async () => { for (const listener of listeners) listener(next); }); },
    async close() { await React.act(async () => { root.unmount(); }); },
  };
}
async function click(text: string) {
  const button = [...document.querySelectorAll("button")].find((node) => node.textContent === text);
  assert.ok(button, text);
  await React.act(async () => { button.click(); });
}

test("failed delivery is separate and retries its result id without submitting Python", async () => {
  const f = setup();
  await f.render();
  assert.match(document.body.textContent!, /Sharing failed/);
  await click("1 failed");
  assert.equal(f.history, 1);
  await click("Retry sharing");
  assert.match(document.body.textContent!, /Shared to team/);
  assert.deepEqual(f.calls.filter((path) => path.endsWith("/retry")), ["/result-sharing/run-1/retry"]);
  assert.ok(f.calls.every((path) => !path.includes("execute")));
  await f.close();
});

test("pending, local-only and read-only statuses do not suggest a false shared result", async () => {
  const f = setup(snapshot("pending"), { readOnly: true });
  await f.render();
  assert.match(document.body.textContent!, /Waiting to share.*1 waiting/);
  assert.equal([...document.querySelectorAll("button")].find((button) => button.textContent === "Retry sharing")!.disabled, true);
  await f.emit(snapshot("local_only"));
  assert.match(document.body.textContent!, /Local only/);
  assert.ok(!document.body.textContent!.includes("Retry sharing"));
  await f.close();
});

test("a shared summary does not claim that a local graph was uploaded", async () => {
  const f = setup(snapshot("shared"));
  await f.render({ ...f.props, partial: true } as typeof f.props);
  assert.match(document.body.textContent!, /Summary shared · graph local/);
  assert.ok(!document.body.textContent!.includes("Shared to team"));
  await f.close();
});

test("historical authorization failures do not disable a recovered session", async () => {
  const f = setup(snapshot("failed", 403));
  await f.render();
  assert.equal(f.denied, 0);
  await f.emit({ ...snapshot("failed", 403), error: { code: "ROLE_REQUIRED", status: 403, message: "Reopen the project to verify access." } });
  assert.equal(f.denied, 1);
  await f.close();
});

test("current global denial is shown ahead of an older selected result's delivery error", async () => {
  const value = { ...snapshot("failed", 409), error: { code: "ROLE_REQUIRED", status: 403,
    message: "Current team access has been revoked." } };
  const f = setup(value);
  await f.render();
  assert.equal(f.denied, 1);
  assert.match(document.body.textContent!, /Current team access has been revoked/);
  assert.ok(!document.body.textContent!.includes("Saved here; sharing needs attention."));
  await f.close();
});

test("status reads wait until project access has been verified by bootstrap", async () => {
  const f = setup(snapshot("failed", 403));
  await f.render({ ...f.props, ready: false });
  assert.equal(f.calls.length, 0);
  assert.equal(f.listeners.size, 0);
  await f.render({ ...f.props, ready: true });
  assert.equal(f.denied, 0);
  assert.match(document.body.textContent!, /Sharing failed/);
  await f.close();
});

test("retry reports revoked access and retains the computed result", async () => {
  const f = setup(snapshot("failed"), { onRetry: async () => { throw new ApiError("Access revoked", 403); } });
  await f.render();
  await click("Retry sharing");
  assert.match(document.body.textContent!, /Sharing failed.*Access revoked/);
  assert.equal(f.denied, 1);
  assert.ok(f.calls.every((path) => !path.includes("execute")));
  await f.close();
});

test("an old project's delayed retry cannot overwrite the newly opened project", async () => {
  const gate = deferred<void>();
  const f = setup(snapshot("failed"), { onRetry: () => gate.promise });
  await f.render();
  const button = [...document.querySelectorAll("button")].find((node) => node.textContent === "Retry sharing")!;
  await React.act(async () => { button.click(); });
  const second = { ...f.client, projectId: "b".repeat(32), request: async <T>() => snapshot("local_only") as T,
    subscribeResultSharing: () => () => {} };
  await f.render({ ...f.props, client: second });
  await React.act(async () => { gate.resolve(); });
  assert.match(document.body.textContent!, /Local only/);
  assert.ok(!document.body.textContent!.includes("Shared to team"));
  await f.close();
  assert.equal(f.listeners.size, 0);
});

test("a delayed retry status read cannot overwrite a newer same-project subscription", async () => {
  const gate = deferred<ResultSharingSnapshot>();
  const f = setup(snapshot("failed"), { retrySnapshot: snapshot("pending"),
    onRead: async (count, value) => count === 2 ? gate.promise : value });
  await f.render();
  await click("Retry sharing");
  assert.match(document.body.textContent!, /Waiting to share/);
  await f.emit(snapshot("shared"));
  assert.match(document.body.textContent!, /Shared to team/);
  await React.act(async () => { gate.resolve(snapshot("pending")); });
  assert.match(document.body.textContent!, /Shared to team/);
  assert.ok(!document.body.textContent!.includes("Waiting to share"));
  await f.close();
});

test("global denial in an accepted retry status read disables the current session", async () => {
  const denied = { ...snapshot("failed"), error: { code: "ROLE_REQUIRED", status: 403, message: "Current access removed." } };
  const f = setup(snapshot("failed"), { retrySnapshot: snapshot("pending"),
    onRead: async (count, value) => count === 2 ? denied : value });
  await f.render();
  await click("Retry sharing");
  assert.equal(f.denied, 1);
  assert.match(document.body.textContent!, /Current access removed/);
  await f.close();
});
