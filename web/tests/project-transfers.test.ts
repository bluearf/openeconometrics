import assert from "node:assert/strict";
import test from "node:test";
import { ApiError } from "../src/api.ts";
import {
  createProjectTransferActions,
  validateProjectTransfer,
  transferErrorMessage,
  type ProjectTransfer,
  type ProjectTransferDependencies,
} from "../src/project-transfers.ts";
import type { DatasetProfile } from "../src/types.ts";
const id = "a".repeat(32),
  cloudId = "b".repeat(32),
  sha = "c".repeat(64);
const view: ProjectTransfer = {
  request_id: id,
  name: "large.csv",
  size_bytes: 9_000_000,
  sha256: sha,
  state: "waiting",
  acknowledged_bytes: 4_000_000,
};
const file = {
  id: cloudId,
  name: view.name,
  data_hash: sha,
  size_bytes: view.size_bytes,
  transfer: "chunked-v1",
};
const imported = {
  id: "33333333-3333-4333-8333-333333333333",
  cloud_id: cloudId,
  sha256: sha,
  data_hash: "d".repeat(64),
  name: view.name,
  row_count: 35,
  column_count: 2,
  columns: [],
  preview: [],
  source: "upload",
  created_at: "2026-10-07",
  size_bytes: 400,
} as DatasetProfile;
function deferred<T>() {
  let resolve!: (x: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
async function fixture(overrides: Partial<ProjectTransferDependencies> = {}) {
  const calls: string[] = [],
    imports: unknown[] = [];
  let role: "editor" | "viewer" | "denied" = "editor";
  const deps: ProjectTransferDependencies = {
    assertAccess(editing) {
      if (role === "denied" || (editing && role === "viewer"))
        throw new ApiError("denied", 403);
    },
    async invoke(action, requestId) {
      calls.push(`${action}:${requestId || ""}`);
      if (action === "list")
        return { status: 200, body: { transfers: [view] } };
      if (action === "status") return { status: 200, body: view };
      if (action === "cancel")
        return { status: 200, body: { cancelled: true, request_id: id } };
      return { status: 200, body: { state: "ready", file } };
    },
    async importReady(value) {
      imports.push(value);
      return imported;
    },
    ...overrides,
  };
  const actions = createProjectTransferActions(deps);
  await actions.refresh();
  return {
    actions,
    calls,
    imports,
    setRole(value: typeof role) {
      role = value;
    },
  };
}
test("list exposes bounded public pins, no token or path, and never resumes", async () => {
  const h = await fixture();
  assert.deepEqual(h.calls, ["list:"]);
  assert.deepEqual(h.actions.getSnapshot().transfers, [view]);
  const snapshot = h.actions.getSnapshot();
  (snapshot.transfers[0] as ProjectTransfer).name = "mutated";
  assert.equal(h.actions.getSnapshot().transfers[0].name, view.name);
  assert.deepEqual(h.imports, []);
});
test("strict transfer view rejects secrets, invalid identifiers, changed size and excess budget", () => {
  for (const patch of [
    { token: "never-forward" },
    { local_path: "/private/file.csv" },
    { request_id: "../escape" },
    { sha256: "short" },
    { size_bytes: 2 * 1024 ** 3 + 1 },
    { acknowledged_bytes: view.size_bytes + 1 },
    { name: "../file.csv" },
    { state: "complete" },
  ])
    assert.throws(
      () => validateProjectTransfer({ ...view, ...patch }),
      /verified/,
    );
});
test("explicit ready resume checks byte pins and imports a separate local dataset profile", async () => {
  const h = await fixture();
  assert.deepEqual(await h.actions.resume(id), imported);
  assert.deepEqual(h.imports, [file]);
  assert.equal(h.actions.getSnapshot().transfers.length, 0);
  assert.notEqual(imported.id, cloudId);
  assert.notEqual(imported.data_hash, sha);
});
test("202 resume remains pending with updated byte progress, and permits explicit retry", async () => {
  let attempts = 0;
  const h = await fixture({
    async invoke(action) {
      if (action === "list")
        return { status: 200, body: { transfers: [view] } };
      attempts++;
      return attempts === 1
        ? { status: 202, body: { ...view, acknowledged_bytes: 8_000_000 } }
        : { status: 200, body: { state: "ready", file } };
    },
  });
  assert.equal(await h.actions.resume(id), null);
  assert.equal(
    h.actions.getSnapshot().transfers[0].acknowledged_bytes,
    8_000_000,
  );
  assert.equal(h.actions.getSnapshot().busyRequestId, null);
  assert.deepEqual(h.imports, []);
  assert.deepEqual(await h.actions.resume(id), imported);
});
test("viewer may inspect but cannot resume or cancel; denied project cannot inspect", async () => {
  const h = await fixture();
  h.setRole("viewer");
  await h.actions.refresh();
  await assert.rejects(h.actions.resume(id), /denied/);
  await assert.rejects(h.actions.cancel(id), /denied/);
  assert.deepEqual(h.calls, ["list:", "list:"]);
  h.setRole("denied");
  await assert.rejects(h.actions.refresh(), /denied/);
  assert.match(h.actions.getSnapshot().error, /permission/);
});
test("closed workspace refuses transport and ignores late ready completion", async () => {
  const pending = deferred<{ status: number; body: unknown }>();
  const h = await fixture({
    async invoke(action) {
      return action === "list"
        ? { status: 200, body: { transfers: [view] } }
        : pending.promise;
    },
  });
  const next = h.actions.resume(id);
  h.actions.dispose();
  pending.resolve({ status: 200, body: { state: "ready", file } });
  await assert.rejects(next, /Project closed/);
  await assert.rejects(h.actions.refresh(), /Project closed/);
  assert.deepEqual(h.imports, []);
});
test("collision in resumed catalogue pins or cache readback never clears pending transfer", async () => {
  for (const badFile of [
    { ...file, data_hash: "f".repeat(64) },
    { ...file, local_path: "private" },
    { ...file, name: "changed.csv" },
    { ...file, transfer: "legacy" },
  ]) {
    const h = await fixture({
      async invoke(action) {
        return action === "list"
          ? { status: 200, body: { transfers: [view] } }
          : { status: 200, body: { state: "ready", file: badFile } };
      },
    });
    await assert.rejects(h.actions.resume(id), /verified/);
    assert.deepEqual(h.imports, []);
    assert.equal(h.actions.getSnapshot().transfers.length, 1);
  }
  const h = await fixture({
    async importReady() {
      return { ...imported, sha256: "f".repeat(64) } as DatasetProfile;
    },
  });
  await assert.rejects(h.actions.resume(id), /verified/);
  assert.equal(h.actions.getSnapshot().transfers.length, 1);
});
test("cancel during active resume retains 202 cancellation and suppresses late upload success", async () => {
  const pending = deferred<{ status: number; body: unknown }>();
  let cancelled = 0;
  const h = await fixture({
    async invoke(action) {
      if (action === "list")
        return { status: 200, body: { transfers: [view] } };
      if (action === "resume") return pending.promise;
      if (action === "cancel")
        return ++cancelled === 1
          ? { status: 202, body: { ...view, state: "cancel_pending" } }
          : { status: 200, body: { cancelled: true, request_id: id } };
      return { status: 200, body: view };
    },
  });
  const next = h.actions.resume(id);
  await h.actions.cancel(id);
  assert.equal(h.actions.getSnapshot().transfers[0].state, "cancel_pending");
  await h.actions.status(id);
  await h.actions.refresh();
  assert.equal(h.actions.getSnapshot().transfers[0].state, "cancel_pending");
  pending.resolve({ status: 200, body: { state: "ready", file } });
  assert.equal(await next, null);
  assert.deepEqual(h.imports, []);
  await assert.rejects(h.actions.resume(id), /Finish or cancel/);
  await h.actions.cancel(id);
  assert.equal(h.actions.getSnapshot().transfers.length, 0);
});
test("cancel offline remains retryable and double cancellation makes one native call", async () => {
  const pending = deferred<{ status: number; body: unknown }>();
  let calls = 0;
  const h = await fixture({
    async invoke(action) {
      if (action === "list")
        return { status: 200, body: { transfers: [view] } };
      calls++;
      return pending.promise;
    },
  });
  const a = h.actions.cancel(id),
    b = h.actions.cancel(id);
  pending.resolve({ status: 503, body: {} });
  await assert.rejects(a);
  await b;
  assert.equal(calls, 1);
  assert.equal(h.actions.getSnapshot().transfers[0].state, "cancel_pending");
  assert.match(h.actions.getSnapshot().error, /Connection lost/);
});
test("a stale list arriving after cancellation cannot resurrect a removed journal", async () => {
  const pending = deferred<{ status: number; body: unknown }>();
  let listed = 0;
  const h = await fixture({
    async invoke(action) {
      if (action === "list")
        return ++listed === 1
          ? { status: 200, body: { transfers: [view] } }
          : pending.promise;
      return { status: 200, body: { cancelled: true, request_id: id } };
    },
  });
  const refreshing = h.actions.refresh();
  await h.actions.cancel(id);
  pending.resolve({ status: 200, body: { transfers: [view] } });
  await refreshing;
  assert.equal(h.actions.getSnapshot().transfers.length, 0);
});
test("a second large transfer cannot be resumed while the first is active", async () => {
  const second = { ...view, request_id: "e".repeat(32) };
  const pending = deferred<{ status: number; body: unknown }>();
  const h = await fixture({
    async invoke(action) {
      return action === "list"
        ? { status: 200, body: { transfers: [view, second] } }
        : pending.promise;
    },
  });
  const first = h.actions.resume(id);
  await assert.rejects(h.actions.resume(second.request_id), /Finish or cancel/);
  pending.resolve({ status: 202, body: view });
  await first;
});
test("list rejects duplicate journals, more than four, and changed source pins", async () => {
  const actions = createProjectTransferActions({
    assertAccess() {},
    async importReady() {
      return imported;
    },
    async invoke() {
      return { status: 200, body: { transfers: [view, view] } };
    },
  });
  await assert.rejects(actions.refresh(), /verified/);
  const tooMany = createProjectTransferActions({
    assertAccess() {},
    async importReady() {
      return imported;
    },
    async invoke() {
      return { status: 200, body: { transfers: Array(5).fill(view) } };
    },
  });
  await assert.rejects(tooMany.refresh(), /verified/);
  let lists = 0;
  const h = await fixture({
    async invoke() {
      return {
        status: 200,
        body: {
          transfers: [
            { ...view, name: ++lists === 1 ? view.name : "changed.csv" },
          ],
        },
      };
    },
  });
  await assert.rejects(h.actions.refresh(), /verified/);
});
test("untrusted native errors are never rendered as credential or local path text", () => {
  assert.ok(
    !transferErrorMessage(
      new Error("Bearer private-token /Users/private/data"),
    ).includes("private"),
  );
  assert.match(
    transferErrorMessage(new ApiError("private", 403)),
    /permission/,
  );
  assert.match(transferErrorMessage("OPENECON_DATA_INTEGRITY"), /verified/);
});
test("throwing detached UI subscriber does not fail the durable transfer", async () => {
  const h = await fixture();
  let observed = 0;
  h.actions.subscribe(() => {
    if (++observed > 1) throw new Error("detached");
  });
  assert.deepEqual(await h.actions.resume(id), imported);
  assert.equal(h.actions.getSnapshot().transfers.length, 0);
});

test("already-published cancellation refusal preserves the journal and permits explicit ready import", async () => {
  const h = await fixture({
    async invoke(action) {
      if (action === "list")
        return { status: 200, body: { transfers: [view] } };
      if (action === "cancel")
        return {
          status: 409,
          body: { detail: { code: "TRANSFER_COMPLETE", message: "private" } },
        };
      return { status: 200, body: { state: "ready", file } };
    },
  });
  await assert.rejects(
    h.actions.cancel(id),
    (error) => error instanceof ApiError && error.code === "TRANSFER_COMPLETE",
  );
  assert.equal(h.actions.getSnapshot().transfers[0].state, "ready");
  assert.match(h.actions.getSnapshot().error, /already complete/);
  assert.ok(!h.actions.getSnapshot().error.includes("private"));
  assert.deepEqual(await h.actions.resume(id), imported);
});
test("cleanup pending structured error keeps a cancellation retry visible", async () => {
  const h = await fixture({
    async invoke(action) {
      return action === "list"
        ? { status: 200, body: { transfers: [view] } }
        : { status: 409, body: { detail: { code: "CLEANUP_PENDING" } } };
    },
  });
  await assert.rejects(h.actions.cancel(id));
  assert.equal(h.actions.getSnapshot().transfers[0].state, "cancel_pending");
  assert.match(h.actions.getSnapshot().error, /retry cleanup/);
});
