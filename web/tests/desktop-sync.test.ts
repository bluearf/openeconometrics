import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { ApiError, submitWorkspaceExecution, type WorkspaceClient } from "../src/api.ts";
import { cloudArchiveRecord, createHybridWorkspace } from "../src/desktop-sync.ts";
import type { ConsoleOutput, DatasetProfile, ExecutionRecord } from "../src/types.ts";
import type { ResultSharingRecord } from "../src/result-sharing.ts";

const localData = { id: "33333333-3333-4333-8333-333333333333", name: "large.csv", data_hash: "d".repeat(64), local_only: true } as DatasetProfile & { local_only: boolean };

const pid = "1".repeat(32);
function fixture() {
  const localCalls: { path: string; body?: Record<string, unknown> }[] = [];
  const cloudCalls: { path: string; body?: Record<string, unknown> }[] = [];
  let code: string | null = null;
  let state: { cloud_version: number; base_code: string; role: "owner" | "editor" | "viewer" } | null = null;
  const outbox: Record<string, unknown>[] = [];
  const sharing = new Map<string, ResultSharingRecord>();
  let online = true, denied = false, version = 0, remoteCode = "a = 1";
  let delaySave: Promise<void> | null = null;
  let cacheFailure: unknown = null;
  let readOnly = false;
  let datasets: DatasetProfile[] = [];
  let selected: DatasetProfile | null = null;
  let chooserCalls = 0;
  let executionOutputs: ConsoleOutput[] = [];
  const session = { token: "local-only", version: "0.3", environment: "local" as const, persistent: true };
  const console = { history: [], variables: [], status: { running: false, session_generation: 1 } };
  const local: WorkspaceClient = {
    projectId: null, teams: false, connect: async () => session, cancelPending() {},
    async request<T>(path, options = {}) {
      const body = options.body ? JSON.parse(String(options.body)) : undefined;
      localCalls.push({ path, body });
      let result: unknown;
      if (path === "/desktop-sync-state") {
        if (options.method === "PUT") state = body;
        result = state;
      } else if (path === "/console/script") {
        if (options.method === "PUT") code = body.code;
        result = { code, name: "analysis.py" };
      } else if (path === "/console") result = console;
      else if (path === "/datasets") result = { datasets };
      else if (path.startsWith("/datasets/")) result = datasets.find((file) => path === `/datasets/${file.id}`);
      else if (path === "/desktop-cached-files") result = { files: [{ cloud_id: "2".repeat(32), sha256: "b".repeat(64) }] };
      else if (path === "/desktop-outbox") {
        if (options.method === "PUT") outbox.push(body);
        result = { items: [...outbox] };
      } else if (path.startsWith("/desktop-outbox/")) {
        const id = decodeURIComponent(path.slice("/desktop-outbox/".length).split("?")[0]);
        const index = outbox.findIndex((item) => (item.record as ExecutionRecord).id === id);
        const item = index >= 0 ? outbox.splice(index, 1)[0] : null;
        const actor_uid = sharing.get(id)?.actor_uid ?? (item?.record as ExecutionRecord)?.actor_uid ?? "qa-editor";
        sharing.set(id, { id, actor_uid, state: "shared", retryable: false });
        result = { deleted: true };
      } else if (path === "/desktop-sharing") {
        for (const item of outbox) {
          const record = item.record as ExecutionRecord;
          if (!sharing.has(record.id)) sharing.set(record.id, { id: record.id, actor_uid: record.actor_uid!, state: "pending", retryable: true });
        }
        result = { records: [...sharing.values()], pending_count: 0, failed_count: 0 };
      } else if (path.startsWith("/desktop-sharing/")) {
        const id = decodeURIComponent(path.slice("/desktop-sharing/".length));
        sharing.set(id, { id, actor_uid: body.actor_uid, state: body.state, error: body.error, retryable: true });
        result = {};
      } else if (path === "/desktop-console/execute") {
        const record = { id: "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa", actor_uid: body.actor_uid, code: body.code, status: "ok", outputs: executionOutputs,
          stdout: "", variables: [], sharing_context: { actor_uid: body.actor_uid, input_files: body.input_files } };
        const localOnly = datasets.some((file) => (file as typeof localData).local_only);
        if (!localOnly) outbox.push({ record, input_files: body.input_files });
        sharing.set(record.id, { id: record.id, actor_uid: body.actor_uid, state: localOnly ? "local_only" : "pending", retryable: !localOnly });
        result = { ...record, ...(localOnly ? { local_only: true } : {}) };
      }
      else throw new Error(`Unexpected local path ${path}`);
      return structuredClone(result) as T;
    },
  };
  const client = createHybridWorkspace(pid, "editor", {
    local, actorUid: "qa-editor",
    async cloud<T>(path, options = {}) {
      const body = options.body ? JSON.parse(String(options.body)) : undefined;
      cloudCalls.push({ path, body });
      if (denied) throw new ApiError("removed", 403);
      if (!online) throw new TypeError("offline");
      let result: unknown;
      if (path === "/bootstrap") result = { session: { read_only: readOnly }, draft: { code: remoteCode, name: "analysis.py", version }, datasets: [], status: console.status };
      else if (path === "/console/script") {
        if (delaySave) { const wait = delaySave; delaySave = null; await wait; }
        if (body.version !== version) throw new ApiError("conflict", 409, "VERSION_CONFLICT");
        remoteCode = body.code; version++;
        result = { code: remoteCode, name: "analysis.py", version };
      } else if (path === "/desktop/results") {
        if ((body.record.outputs as ConsoleOutput[]).some((output) => output.type === "plot" &&
            output.data && typeof output.data === "object" && "artifact" in output.data))
          throw new ApiError("Workspace-local graph artifacts cannot be uploaded.", 422, "INVALID_RESULT");
        result = { shared: true, id: createHash("sha256").update(`qa-editor:${body.record.id}`).digest("hex").slice(0, 32) };
      }
      else if (path.startsWith("/console/history?")) result = { runs: [{ id: "shared-run" }], next_cursor: null };
      else if (path === "/runs/shared-run/record") result = { id: "shared-run", code: "print(42)", outputs: [], status: "ok" };
      else throw new Error(`Unexpected cloud path ${path}`);
      return result as T;
    },
    cacheFile: async () => { throw new Error("No fixture files"); },
    cacheFiles: async () => { if (cacheFailure) throw cacheFailure; return []; },
    chooseAndUpload: async () => { chooserCalls++; if (selected) datasets.push(selected); return selected; },
  });
  return {
    client, localCalls, cloudCalls, outbox,
    get code() { return code; }, get state() { return state; },
    offline() { online = false; }, online() { online = true; }, deny() { denied = true; },
    remoteChange() { remoteCode = "other = 2"; version++; },
    localChange(value: string) { code = value; },
    failCache(error: unknown) { cacheFailure = error; },
    downgrade() { readOnly = true; },
    select(file: DatasetProfile | null) { selected = file; },
    persistDataset(file: DatasetProfile) { datasets.push(file); },
    outputs(value: ConsoleOutput[]) { executionOutputs = value; },
    get chooserCalls() { return chooserCalls; },
    delayNextSave() {
      let release!: () => void;
      delaySave = new Promise<void>((resolve) => { release = resolve; });
      return release;
    },
  };
}

test("desktop execute goes only to the local worker and removes cloud dispatch flag", async () => {
  const f = fixture(); await f.client.request("/bootstrap");
  const record = await submitWorkspaceExecution(f.client, "print(42)");
  assert.equal("status" in record && record.status, "ok");
  const run = f.localCalls.find((c) => c.path === "/desktop-console/execute")!;
  assert.deepEqual(run.body, { code: "print(42)", actor_uid: "qa-editor", input_files: [] });
  assert.equal(f.cloudCalls.some((c) => c.path === "/console/execute"), false);
});

test("offline autosave preserves exact local text and marks pending synchronization", async () => {
  const f = fixture(); await f.client.request("/bootstrap"); f.offline();
  const saved = await f.client.request<{ pending_sync: boolean }>("/console/script", { method: "PUT", body: JSON.stringify({ code: "df = 42" }) });
  assert.equal(saved.pending_sync, true); assert.equal(f.code, "df = 42");
  assert.equal(f.state?.base_code, "a = 1");
});

test("successful upload advances optimistic draft version only after cloud acknowledgement", async () => {
  const f = fixture(); await f.client.request("/bootstrap");
  const saved = await f.client.request<{ version: number; pending_sync: boolean }>("/console/script", { method: "PUT", body: JSON.stringify({ code: "x = 9", version: 999 }) });
  assert.equal(saved.version, 1); assert.equal(saved.pending_sync, false);
  assert.equal(f.state?.base_code, "x = 9");
  assert.equal(f.cloudCalls.at(-1)?.body?.version, 0);
});

test("reopening never overwrites an offline draft with the cloud base", async () => {
  const f = fixture(); await f.client.request("/bootstrap"); f.localChange("pending = 5");
  const boot = await f.client.request<{ draft: { code: string; pending_sync: boolean } }>("/bootstrap");
  assert.equal(boot.draft.code, "pending = 5"); assert.equal(boot.draft.pending_sync, true);
  assert.equal(f.code, "pending = 5");
});

test("a concurrent cloud edit freezes synchronization and preserves the local draft", async () => {
  const f = fixture(); await f.client.request("/bootstrap"); f.localChange("pending = 5"); f.remoteChange();
  await f.client.request("/bootstrap");
  await assert.rejects(f.client.request("/console/script", { method: "PUT", body: JSON.stringify({ code: "pending = 6" }) }), (e) => e instanceof ApiError && e.status === 409);
  assert.equal(f.code, "pending = 6");
  assert.equal(f.cloudCalls.filter((c) => c.path === "/console/script").length, 0);
});

test("a known viewer downgrade persists even when the local draft has a version conflict", async () => {
  const f = fixture(); await f.client.request("/bootstrap");
  f.localChange("pending = 5"); f.remoteChange(); f.downgrade();
  await f.client.request("/bootstrap");
  assert.equal(f.state?.role, "viewer");
  f.offline();
  const boot = await f.client.request<{ session: { read_only: boolean } }>("/bootstrap");
  assert.equal(boot.session.read_only, true);
  assert.equal(f.code, "pending = 5");
});

test("viewer reopening reads result-sharing metadata without a false access-denied publication", async () => {
  const f = fixture(); await f.client.request("/bootstrap");
  const snapshots: { error?: { status?: number } }[] = [];
  f.client.subscribeResultSharing?.((value) => { snapshots.push(value); });
  f.downgrade(); await f.client.request("/bootstrap");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(f.state?.role, "viewer");
  assert.equal(snapshots.some((value) => value.error?.status === 403), false);
  assert.equal(f.cloudCalls.some((call) => call.path === "/desktop/results"), false);
});

test("offline cached project can open, but explicit cloud authorization denial cannot", async () => {
  const f = fixture(); await f.client.request("/bootstrap"); f.offline();
  const boot = await f.client.request<{ draft: { code: string } }>("/bootstrap");
  assert.equal(boot.draft.code, "a = 1"); f.online(); f.deny();
  await assert.rejects(f.client.request("/bootstrap"), (e) => e instanceof ApiError && e.status === 403);
});

test("a first offline checkout fails instead of entering an unauthenticated cached project", async () => {
  const f = fixture(); f.offline();
  await assert.rejects(f.client.request("/bootstrap"));
  assert.equal(f.code, null);
});

test("a connection lost during native file caching uses the previously opened local project", async () => {
  const f = fixture(); await f.client.request("/bootstrap");
  f.failCache(new TypeError("connection lost"));
  const boot = await f.client.request<{ draft: { code: string } }>("/bootstrap");
  assert.equal(boot.draft.code, "a = 1");
  f.failCache(new ApiError("membership removed", 403));
  await assert.rejects(f.client.request("/bootstrap"), (e) => e instanceof ApiError && e.status === 403);
  f.failCache(new Error("corrupt cache"));
  await assert.rejects(f.client.request("/bootstrap"), /corrupt cache/);
});

test("offline result publication retains the outbox and does not rerun Python", async () => {
  const f = fixture(); await f.client.request("/bootstrap"); f.offline();
  await f.client.request("/bootstrap");
  await submitWorkspaceExecution(f.client, "42");
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(f.outbox.length, 1);
  assert.equal((f.outbox[0].record as {actor_uid:string}).actor_uid, "qa-editor");
  assert.deepEqual(f.outbox[0].input_files, [{ id: "2".repeat(32), data_hash: "b".repeat(64) }]);
  assert.equal(f.localCalls.filter((c) => c.path === "/desktop-console/execute").length, 1);
});

test("pending results from another signed-in account are never shared as the current user", async () => {
  const f = fixture();
  f.outbox.push({ record: { id: "other-run", actor_uid: "other-account" }, input_files: [] });
  await f.client.request("/bootstrap");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(f.cloudCalls.some((call) => call.path === "/desktop/results"), false);
  assert.equal(f.outbox.length, 1);
});

test("reconnection queues behind pending saves without creating a false collaborator conflict", async () => {
  const previousWindow = globalThis.window;
  globalThis.window = new EventTarget() as unknown as Window & typeof globalThis;
  const f = fixture();
  try {
    await f.client.request("/bootstrap");
    const release = f.delayNextSave();
    const first = f.client.request("/console/script", { method: "PUT", body: JSON.stringify({ code: "first = 1" }) });
    await new Promise((resolve) => setTimeout(resolve, 0));
    const second = f.client.request("/console/script", { method: "PUT", body: JSON.stringify({ code: "second = 2" }) });
    window.dispatchEvent(new Event("online"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    release();
    await Promise.all([first, second]);
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(f.code, "second = 2");
    assert.equal(f.state?.base_code, "second = 2");
    assert.equal(f.state?.cloud_version, 2);
    assert.equal(f.cloudCalls.filter((call) => call.path === "/console/script").length, 2);
  } finally {
    f.client.cancelPending();
    if (previousWindow) globalThis.window = previousWindow;
    else delete (globalThis as { window?: unknown }).window;
  }
});

test("native local-only import bypasses cloud caching and survives online and offline reopening", async () => {
  const f = fixture(); await f.client.request("/bootstrap"); f.select(localData);
  const uploaded = await f.client.request<DatasetProfile>("/datasets/upload", { method: "POST" });
  assert.equal(uploaded.id, localData.id);
  assert.equal(f.localCalls.some((call) => call.path === "/datasets/import-cached"), false);
  for (const offline of [false, true]) {
    if (offline) f.offline();
    const boot = await f.client.request<{ datasets: DatasetProfile[] }>("/bootstrap");
    assert.deepEqual(boot.datasets, [localData]);
  }
  assert.equal(f.cloudCalls.some((call) => call.path.includes("datasets/")), false);
});

test("runs with persistent local-only inputs retain local execution without a false cloud result", async () => {
  const f = fixture(); f.persistDataset(localData); await f.client.request("/bootstrap");
  const record = await submitWorkspaceExecution(f.client, "print(42)");
  assert.equal((record as { local_only?: boolean }).local_only, true);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(f.outbox.length, 0);
  assert.equal(f.cloudCalls.some((call) => call.path === "/desktop/results"), false);
  assert.equal(f.localCalls.filter((call) => call.path === "/desktop-console/execute").length, 1);
  const saved = await f.client.request<{ pending_sync: boolean }>("/console/script", {
    method: "PUT", body: JSON.stringify({ code: "shared_code = 1" }),
  });
  assert.equal(saved.pending_sync, false); // Other project/code synchronization continues.
});

test("viewer and revoked projects cannot open the native file picker", async () => {
  for (const revoked of [false, true]) {
    const f = fixture(); await f.client.request("/bootstrap"); f.select(localData);
    if (revoked) { f.deny(); await assert.rejects(f.client.request("/bootstrap")); }
    else { f.downgrade(); await f.client.request("/bootstrap"); }
    await assert.rejects(f.client.request("/datasets/upload", { method: "POST" }), (error) => error instanceof ApiError && error.status === 403);
    assert.equal(f.chooserCalls, 0);
  }
});

test("local native record is independently re-read and cannot masquerade as a different input", async () => {
  const f = fixture(); await f.client.request("/bootstrap");
  f.select({ ...localData, id: "not-a-local-uuid" });
  await assert.rejects(f.client.request("/datasets/upload", { method: "POST" }), /local dataset identifier/);
});


function lazyOutput(): ConsoleOutput {
  return { type: "plot", data: { kind: "network", title: "Study network", sample_n: 100000,
    total_n: 100000, artifact: { version: 1, id: "f".repeat(64), bytes: 54000000 } },
    latex: "\\begin{tabular}{lr}Nodes & 100000\\end{tabular}", latex_math: null,
    latex_style: "publication-v1" };
}

function record(id: string, outputs: ConsoleOutput[]): ExecutionRecord {
  const hash = createHash("sha256").update(id).digest("hex");
  return { id: `${hash.slice(0, 8)}-${hash.slice(8, 12)}-4${hash.slice(13, 16)}-a${hash.slice(17, 20)}-${hash.slice(20, 32)}`,
    actor_uid: "qa-editor", code: "print('before'); display(graph); print('after')",
    status: "ok", stdout: "before\nafter\n", error: null, outputs, variables: [], duration_ms: 2,
    session_generation: 1, events: [{ type: "stdout", text: "before\n" },
      { type: "output", index: 0 }, { type: "stdout", text: "after\n" }] };
}

test("cloud graph summaries preserve complete local record, LaTeX and ordered events", () => {
  const source = record("network-run", [lazyOutput(), { type: "text", data: "other output" }]);
  const before = structuredClone(source);
  const archived = cloudArchiveRecord(source);
  assert.equal(archived.local_plot_count, 1);
  assert.equal(archived.record.outputs[0].type, "text");
  assert.match(String(archived.record.outputs[0].data), /chart was not uploaded/);
  assert.match(String(archived.record.outputs[0].data), /HTML/);
  assert.equal(archived.record.outputs[0].latex, source.outputs[0].latex);
  assert.equal(archived.record.outputs[0].latex_math, null);
  assert.equal(archived.record.outputs[0].latex_style, "publication-v1");
  assert.deepEqual(archived.record.events, source.events);
  assert.deepEqual(archived.record.outputs[1], source.outputs[1]);
  assert.deepEqual(source, before);
  assert.equal(JSON.stringify(archived.record).includes("artifact"), false);
});

test("a lazy network result is archived explicitly as partial and does not poison subsequent outbox results", async () => {
  const f = fixture();
  f.outbox.push({ record: record("graph-run", [lazyOutput()]), input_files: [] });
  f.outbox.push({ record: record("ordinary-run", [{ type: "text", data: "42" }]), input_files: [] });
  await f.client.request("/bootstrap");
  // WebCrypto acknowledgement checks can outlive a 10 ms sleep under a full
  // parallel test run. Wait for durable queue completion, without retrying it.
  const deadline = Date.now() + 5000;
  while (f.outbox.length && Date.now() < deadline)
    await new Promise((resolve) => setTimeout(resolve, 10));
  const publications = f.cloudCalls.filter((call) => call.path === "/desktop/results");
  assert.equal(publications.length, 2);
  const first = publications[0].body!.record as ExecutionRecord;
  assert.equal(first.outputs[0].type, "text");
  assert.match(String(first.outputs[0].data), /chart was not uploaded/);
  assert.equal((publications[1].body!.record as ExecutionRecord).outputs[0].data, "42");
  assert.equal(f.outbox.length, 0);
  assert.equal(f.localCalls.some((call) => call.path === "/desktop-console/execute"), false);
});

test("an offline lazy graph keeps its complete reference and marks partial sharing after execution", async () => {
  const f = fixture(); await f.client.request("/bootstrap"); f.offline();
  const output = lazyOutput(); f.outputs([output]);
  const returned = await submitWorkspaceExecution(f.client, "display(graph)") as ExecutionRecord & { sharing_scope?: string; local_plot_count?: number };
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(returned.sharing_scope, "partial");
  assert.equal(returned.local_plot_count, 1);
  assert.deepEqual(returned.outputs, [output]);
  assert.equal(f.outbox.length, 1);
  assert.deepEqual((f.outbox[0].record as ExecutionRecord).outputs, [output]);
});


test("shared history and saved results use cloud reads without changing local console history", async () => {
  const f = fixture(); await f.client.request("/bootstrap");
  // Background sharing status reads may finish while browsing. Only local
  // console/history operations could replace a local run or execute Python.
  const consoleCalls = () => f.localCalls.filter(call => call.path === "/console" ||
    call.path.startsWith("/console/") || call.path.startsWith("/runs/") ||
    call.path === "/desktop-console/execute");
  const before = [...consoleCalls()];
  const page = await f.client.request<{runs:{id:string}[]}>("/console/history?query=print&limit=20");
  const record = await f.client.request<{id:string}>("/runs/shared-run/record");
  assert.equal(page.runs[0].id, "shared-run"); assert.equal(record.id, "shared-run");
  assert.deepEqual(consoleCalls(), before);
  await assert.rejects(f.client.request("/console/history?limit=20", {method:"POST"}), error => error instanceof ApiError && error.status === 405);
  await assert.rejects(f.client.request("/runs/shared-run/record", {method:"PUT"}), error => error instanceof ApiError && error.status === 405);
  assert(!f.cloudCalls.some(call => call.path === "/console/execute"));
});

test("shared history propagates offline and revoked access failures without local fallback", async () => {
  const f = fixture(); await f.client.request("/bootstrap"); f.offline();
  await assert.rejects(f.client.request("/console/history?limit=20"), TypeError);
  f.online(); f.deny();
  await assert.rejects(f.client.request("/console/history?limit=20"), error => error instanceof ApiError && error.status === 403);
});
