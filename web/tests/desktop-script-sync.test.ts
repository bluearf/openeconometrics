import { test } from "node:test";
import assert from "node:assert/strict";
import { ApiError, type ScriptFile, type ScriptSummary, type ScriptSyncNotification, type WorkspaceClient } from "../src/api.ts";
import { createHybridWorkspace } from "../src/desktop-sync.ts";
import { createNamedScriptSync, type DesktopSyncState } from "../src/desktop-script-sync.ts";
import { LatestValueAutosaver } from "../src/session-state.ts";
import { VersionedScriptWriter } from "../src/api.ts";

const A = "a".repeat(32), B = "b".repeat(32), C = "c".repeat(32);
const path = (id: string) => `/console/scripts/${id}`;
const put = (code: string, version: number) => ({ method: "PUT", body: JSON.stringify({ code, version }) });
const post = (id: string, name = "file.py", code = "") => ({ method: "POST", body: JSON.stringify({ id, name, code }) });
const isStatus = (status: number) => (error: unknown) => error instanceof ApiError && error.status === status;
const remoteReload = { headers: { "X-OpenEcon-Resolve-Conflict": "remote" } };
async function until(predicate: () => boolean) {
  const deadline = Date.now() + 1500;
  while (!predicate()) {
    assert.ok(Date.now() < deadline, "Expected background synchronization to finish");
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
}

function fixture() {
  const localFiles = new Map<string, ScriptFile>([["analysis", { id: "analysis", name: "analysis.py", code: "", version: 0 }]]);
  const cloudFiles = new Map<string, ScriptFile>([["analysis", { id: "analysis", name: "analysis.py", code: "main = 1", version: 7 }]]);
  const localCalls: { path: string; method: string; body: any }[] = [];
  const cloudCalls: typeof localCalls = [];
  let state: DesktopSyncState | null = null;
  let online = true, denied = false, viewer = false, supportsFiles = true;
  let legacyDocumentNames = false;
  let creationFailure: ApiError | null = null;
  let lostAck: "POST" | "PUT" | null = null;
  let held: { path: string; method: string; response: boolean; entered(): void; wait: Promise<void> } | null = null;
  let localFailure: { path: string; error: Error } | null = null;
  let heldState: { entered(): void; wait: Promise<void> } | null = null;
  const session = { token: "local", version: "0.3", environment: "local" as const, persistent: true };
  const console = { history: [], variables: [], status: { running: false, session_generation: 1 } };
  const meta = (files: Map<string, ScriptFile>) => ({ scripts: [...files.values()].map(({ id, name, version }) => ({ id, name, version })) });
  function get(files: Map<string, ScriptFile>, id: string) {
    const file = files.get(id);
    if (!file) throw new ApiError("missing", 404, "NOT_FOUND");
    return file;
  }
  function create(files: Map<string, ScriptFile>, body: any) {
    const previous = files.get(body.id);
    if (previous) {
      if (previous.name === body.name && previous.code === body.code) return previous;
      throw new ApiError("id conflict", 409, "SCRIPT_ID_CONFLICT");
    }
    const requested = body.name ?? "untitled.py";
    let name = requested, number = 2;
    while ([...files.values()].some((entry) => entry.name === name)) name = `${requested.slice(0, -3)}_${number++}.py`;
    const result = { id: body.id, name, code: body.code ?? "", version: 0 };
    files.set(body.id, result);
    return result;
  }
  function save(files: Map<string, ScriptFile>, id: string, body: any) {
    const previous = get(files, id);
    if (previous.version !== body.version) throw new ApiError("changed", 409, "VERSION_CONFLICT");
    const result = { ...previous, code: body.code, version: previous.version + 1 };
    files.set(id, result);
    return result;
  }
  const local: WorkspaceClient = {
    projectId: null, teams: false, connect: async () => session, cancelPending() {},
    async request<T>(pathname: string, options: RequestInit = {}) {
      const method = options.method ?? "GET", body = options.body ? JSON.parse(String(options.body)) : null;
      localCalls.push({ path: pathname, method, body });
      if (method === "PUT" && localFailure?.path === pathname) {
        const failure = localFailure; localFailure = null; throw failure.error;
      }
      let result: unknown;
      if (pathname === "/desktop-sync-state") {
        if (method === "PUT" && heldState) {
          const pending = heldState; heldState = null; pending.entered(); await pending.wait;
        }
        if (method === "PUT") state = structuredClone(body);
        result = state;
      } else if (pathname === "/console/script") {
        if (method === "PUT") save(localFiles, "analysis", { ...body, version: get(localFiles, "analysis").version });
        const main = get(localFiles, "analysis"); result = { code: main.code, name: main.name };
      } else if (pathname === "/console/scripts") result = method === "POST" ? create(localFiles, body) : meta(localFiles);
      else if (pathname.startsWith("/console/scripts/")) {
        const id = pathname.split("/").at(-1)!;
        result = method === "PUT" ? save(localFiles, id, body) : get(localFiles, id);
      } else if (pathname.startsWith("/desktop-scripts/")) {
        const id = pathname.split("/").at(-1)!, previous = localFiles.get(id);
        if (previous && previous.version !== body.version) throw new ApiError("local conflict", 409, "VERSION_CONFLICT");
        if ([...localFiles.values()].some((entry) => entry.id !== id && entry.name === body.name))
          throw new ApiError("name conflict", 409, "SCRIPT_NAME_CONFLICT");
        result = previous && previous.name === body.name && previous.code === body.code ? previous
          : { id, name: body.name, code: body.code, version: previous ? previous.version + 1 : 0 };
        localFiles.set(id, result as ScriptFile);
      } else if (pathname === "/console") result = console;
      else if (pathname === "/datasets") result = { datasets: [] };
      else if (pathname === "/desktop-cached-files") result = { files: [] };
      else if (pathname === "/desktop-outbox") result = { items: [] };
      else throw new Error(`Unexpected local request: ${pathname}`);
      return structuredClone(result) as T;
    },
  };
  function client() {
    return createHybridWorkspace("1".repeat(32), "editor", {
      actorUid: "editor", local, cacheFile: async () => { throw new Error("No data files"); },
      async cloud<T>(pathname: string, options: RequestInit = {}) {
        const method = options.method ?? "GET", body = options.body ? JSON.parse(String(options.body)) : null;
        cloudCalls.push({ path: pathname, method, body });
        const pending = held?.path === pathname && held.method === method ? held : null;
        if (pending) held = null;
        if (pending && !pending.response) { pending.entered(); await pending.wait; }
        if (denied) throw new ApiError("removed", 403);
        if (!online) throw new TypeError("offline");
        if (viewer && method !== "GET") throw new ApiError("viewer", 403);
        let result: unknown;
        if (pathname === "/bootstrap") result = { session: { read_only: viewer }, draft: get(cloudFiles, "analysis"), datasets: [], status: console.status };
        else if (pathname === "/console/script") result = method === "PUT" ? save(cloudFiles, "analysis", body) : get(cloudFiles, "analysis");
        else if (pathname.startsWith("/console/scripts")) {
          if (!supportsFiles) throw new ApiError("not deployed", 404);
          if (pathname === "/console/scripts" && method === "POST") {
            if (creationFailure) throw creationFailure;
            if (legacyDocumentNames && /\.(md|tex)$/.test(body.name))
              throw new ApiError("Güvenli, kısa bir .py dosya adı kullanın.", 422, "INVALID_SCRIPT");
          }
          result = pathname === "/console/scripts" ? (method === "POST" ? create(cloudFiles, body) : meta(cloudFiles))
            : method === "PUT" ? save(cloudFiles, pathname.split("/").at(-1)!, body) : get(cloudFiles, pathname.split("/").at(-1)!);
        } else throw new Error(`Unexpected cloud request: ${pathname}`);
        if (pending?.response) { pending.entered(); await pending.wait; }
        if (lostAck === method) { lostAck = null; throw new TypeError("response lost"); }
        return structuredClone(result) as T;
      },
    });
  }
  return { client, localFiles, cloudFiles, localCalls, cloudCalls, get state() { return state; },
    offline: () => { online = false; }, online: () => { online = true; },
    deny: () => { denied = true; }, allow: () => { denied = false; },
    viewer: () => { viewer = true; }, oldServer: () => { supportsFiles = false; },
    legacyDocuments: () => { legacyDocumentNames = true; }, documentUpgrade: () => { legacyDocumentNames = false; },
    rejectCreation: (error: ApiError | null) => { creationFailure = error; },
    loseAck: (method: "POST" | "PUT") => { lostAck = method; },
    failLocal: (pathname: string, error = new Error("disk write failed")) => { localFailure = { path: pathname, error }; },
    holdCloud(pathname: string, method = "PUT", response = false) {
      let entered!: () => void, release!: () => void;
      const started = new Promise<void>((resolve) => { entered = resolve; });
      const wait = new Promise<void>((resolve) => { release = resolve; });
      held = { path: pathname, method, response, entered, wait };
      return { entered: started, release };
    },
    holdState() {
      let entered!: () => void, release!: () => void;
      const started = new Promise<void>(resolve => { entered = resolve; });
      const wait = new Promise<void>(resolve => { release = resolve; });
      heldState = { entered, wait }; return { entered: started, release };
    },
    remoteFile: (id = A, name = "remote.py", code = "remote = 1", version = 12) => {
      const file = { id, name, code, version }; cloudFiles.set(id, file); return file;
    },
  };
}

test("empty named file is durable and opening/listing files never executes code", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap");
  const file = await client.request<ScriptFile>("/console/scripts", post(A));
  assert.deepEqual(file, { id: A, name: "file.py", code: "", version: 0, pending_sync: false });
  const files = await client.request<{ scripts: ScriptSummary[] }>("/console/scripts");
  assert.equal(files.scripts[0].version, f.localFiles.get("analysis")!.version);
  assert.equal(files.scripts.some((entry) => "code" in entry), false);
  const reopened = f.client(); await reopened.request("/bootstrap");
  assert.equal((await reopened.request<ScriptFile>(path(A))).code, "");
  assert.equal(f.state?.scripts?.[A].cloud_version, 0);
  assert.equal([...f.localCalls, ...f.cloudCalls].some((call) => call.path.includes("execute")), false);
});

test("offline creation and edits survive reopening; retry sends the same UUID and original POST", async () => {
  const previous = globalThis.window; globalThis.window = new EventTarget() as unknown as Window & typeof globalThis;
  const f = fixture(), first = f.client(); let reopened: WorkspaceClient | undefined;
  try {
    await first.request("/bootstrap"); f.offline();
    const created = await first.request<ScriptFile>("/console/scripts", post(A, "new.py", ""));
    assert.equal(created.pending_sync, true);
    const saved = await first.request<ScriptFile>(path(A), put("answer = 42", created.version));
    assert.equal(saved.version, 1); assert.equal(saved.pending_sync, true);
    first.cancelPending(); reopened = f.client(); await reopened.request("/bootstrap");
    assert.equal((await reopened.request<ScriptFile>(path(A))).code, "answer = 42");
    f.online(); window.dispatchEvent(new Event("online"));
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(f.cloudFiles.get(A)?.code, "answer = 42");
    assert.equal(f.state?.scripts?.[A].cloud_version, 1);
    const creates = f.cloudCalls.filter((call) => call.path === "/console/scripts" && call.method === "POST");
    assert.ok(creates.length >= 2);
    assert.ok(creates.every((call) => call.body.id === A && call.body.code === ""));
  } finally {
    first.cancelPending(); reopened?.cancelPending();
    if (previous) globalThis.window = previous; else delete (globalThis as { window?: unknown }).window;
  }
});

test("local version and cloud version stay independent; stale local writes cannot touch the cloud", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap");
  const first = await client.request<ScriptFile>(path(A));
  assert.equal(first.version, 0); assert.equal(f.state?.scripts?.[A].cloud_version, 12);
  const saved = await client.request<ScriptFile>(path(A), put("changed = 2", 0));
  assert.equal(saved.version, 1); assert.equal(saved.pending_sync, true);
  await until(() => f.state?.scripts?.[A].cloud_version === 13);
  assert.equal(f.cloudFiles.get(A)?.version, 13);
  assert.equal(f.cloudCalls.findLast((call) => call.method === "PUT")?.body.version, 12);
  const previousWrites = f.cloudCalls.filter((call) => call.method === "PUT").length;
  await assert.rejects(client.request(path(A), put("stale = 3", 0)), isStatus(409));
  assert.equal(f.localFiles.get(A)?.code, "changed = 2");
  assert.equal(f.cloudCalls.filter((call) => call.method === "PUT").length, previousWrites);
});

test("conflicts preserve a file's local draft without blocking a different file", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); f.remoteFile(B, "second.py"); await client.request("/bootstrap");
  await client.request(path(A)); await client.request(path(B));
  f.remoteFile(A, "remote.py", "collaborator = 9", 13);
  const pending = await client.request<ScriptFile>(path(A), put("my_draft = 4", 0));
  assert.equal(pending.pending_sync, true);
  await until(() => Boolean(f.state?.scripts?.[A].conflict));
  assert.equal(f.localFiles.get(A)?.code, "my_draft = 4"); assert.equal(f.cloudFiles.get(A)?.code, "collaborator = 9");
  assert.equal(f.state?.scripts?.[A].conflict, true);
  const saved = await client.request<ScriptFile>(path(B), put("independent = 3", 0));
  assert.equal(saved.pending_sync, true);
  await until(() => f.state?.scripts?.[B].base_code === "independent = 3");
  assert.equal(f.cloudFiles.get(B)?.code, "independent = 3");
  await client.request("/bootstrap");
  assert.equal(f.state?.scripts?.[A].conflict, true, "legacy bootstrap must preserve per-file states");
});

test("a remotely updated clean document backfills locally using the local optimistic version", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap");
  await client.request(path(A)); f.remoteFile(A, "remote.py", "cloud = 99", 40);
  const updated = await client.request<ScriptFile>(path(A), remoteReload);
  assert.equal(updated.code, "cloud = 99"); assert.equal(updated.version, 1);
  assert.equal(f.state?.scripts?.[A].cloud_version, 40);
  assert.equal(f.localCalls.findLast((call) => call.path.startsWith("/desktop-scripts/"))?.body.version, 0);
});

test("modern main file uses local CAS while its cloud synchronization retains the legacy draft route", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap");
  const initial = await client.request<ScriptFile>(path("analysis"));
  assert.equal(initial.version, 1);
  const saved = await client.request<ScriptFile>(path("analysis"), put("main = 2", initial.version));
  assert.equal(saved.id, "analysis"); assert.equal(saved.version, 2); assert.equal(saved.pending_sync, true);
  await until(() => f.state?.base_code === "main = 2");
  assert.equal(f.state?.cloud_version, 8);
  assert.equal(f.cloudCalls.some((call) => call.path === path("analysis")), false);
  await assert.rejects(client.request(path("analysis"), put("bad = 3", initial.version)), isStatus(409));
  assert.equal(f.localFiles.get("analysis")?.code, "main = 2");
});

test("viewer writes are rejected before touching any local file", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); f.viewer(); await client.request("/bootstrap");
  await client.request(path(A)); const before = structuredClone([...f.localFiles]);
  await assert.rejects(client.request(path(A), put("forbidden = 1", 0)), isStatus(403));
  await assert.rejects(client.request("/console/scripts", post(B)), isStatus(403));
  await assert.rejects(client.request(path("analysis"), put("forbidden = 2", 7)), isStatus(403));
  assert.deepEqual([...f.localFiles], before);
});

test("explicit authorization loss is persisted and never becomes cached offline access", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  f.deny(); await assert.rejects(client.request("/console/scripts"), isStatus(403));
  assert.equal(f.state?.access_denied, true); f.allow(); f.offline();
  const reopened = f.client(); await assert.rejects(reopened.request("/bootstrap"), isStatus(403));
  const before = structuredClone([...f.localFiles]);
  await assert.rejects(reopened.request(path(A), put("forbidden = 1", 0)), isStatus(403));
  assert.deepEqual([...f.localFiles], before);
  f.online(); await reopened.request("/bootstrap");
  assert.equal(f.state?.access_denied, false);
});

test("an old cloud deployment returns an error before creating an unsynchronized local file", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap"); f.oldServer();
  await assert.rejects(client.request("/console/scripts", post(A)), isStatus(404));
  assert.equal(f.localFiles.has(A), false); assert.equal(f.cloudFiles.has(A), false);
});

test("lost creation acknowledgements retry idempotently without duplicating or overwriting files", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap"); f.loseAck("POST");
  const created = await client.request<ScriptFile>("/console/scripts", post(A, "new.py", "first = 1"));
  assert.equal(created.pending_sync, true); assert.equal(f.state?.scripts?.[A].cloud_version, null);
  const saved = await client.request<ScriptFile>(path(A), put("second = 2", 0));
  assert.equal(saved.pending_sync, true);
  await until(() => f.state?.scripts?.[A].base_code === "second = 2");
  assert.equal(f.cloudFiles.size, 2);
  const creates = f.cloudCalls.filter((call) => call.path === "/console/scripts" && call.method === "POST");
  assert.deepEqual(creates[0].body, creates[1].body);
  assert.equal(f.cloudFiles.get(A)?.code, "second = 2");
});

test("lost PUT acknowledgements reconcile exact cloud content and preserve local version", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  f.loseAck("PUT"); const pending = await client.request<ScriptFile>(path(A), put("saved = 4", 0));
  assert.equal(pending.pending_sync, true);
  await until(() => f.cloudFiles.get(A)?.code === "saved = 4");
  const verified = await client.request<ScriptFile>(path(A), remoteReload);
  assert.equal(verified.pending_sync, false); assert.equal(verified.version, 1);
  assert.equal(f.state?.scripts?.[A].cloud_version, 13);
});

test("same-name creation suffixes safely and a supplied existing ID cannot claim cloud ownership", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(B, "file.py"); await client.request("/bootstrap");
  const created = await client.request<ScriptFile>("/console/scripts", post(A));
  assert.equal(created.name, "file_2.py"); assert.equal(f.localFiles.get(A)?.name, "file_2.py");
  assert.equal(f.cloudFiles.get(B)?.code, "remote = 1");
  await assert.rejects(client.request("/console/scripts", post(B, "file.py", "overwrite")), isStatus(409));
  assert.equal(f.localFiles.has(B), false);
  assert.equal(f.cloudFiles.get(B)?.code, "remote = 1");
});

test("new files omitted from the catalog while offline remain visible on the next catalog read", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap");
  await client.request("/console/scripts"); f.offline();
  await client.request("/console/scripts", post(C, "offline.py", "kept = 1"));
  const list = await client.request<{ scripts: ScriptSummary[] }>("/console/scripts");
  assert.equal(list.scripts.find((file) => file.id === C)?.pending_sync, true);
});

test("explicit named conflict reload imports the remote version; ordinary GET preserves the draft", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  f.remoteFile(A, "remote.py", "other = 9", 13);
  await client.request(path(A), put("local = 2", 0));
  await until(() => Boolean(f.state?.scripts?.[A].conflict));
  assert.equal((await client.request<ScriptFile>(path(A))).code, "local = 2");
  f.offline();
  const reload = { headers: { "X-OpenEcon-Resolve-Conflict": "remote" } };
  await assert.rejects(client.request(path(A), reload));
  assert.equal(f.localFiles.get(A)?.code, "local = 2");
  f.online(); const resolved = await client.request<ScriptFile>(path(A), reload);
  assert.equal(resolved.code, "other = 9"); assert.equal(resolved.pending_sync, false);
  assert.equal(f.state?.scripts?.[A].conflict, undefined);
  const next = await client.request<ScriptFile>(path(A), put("after_reload = 1", resolved.version));
  assert.equal(next.pending_sync, true);
  await until(() => f.state?.scripts?.[A].base_code === "after_reload = 1");
  assert.equal(f.cloudFiles.get(A)?.code, "after_reload = 1");
});

test("explicit analysis conflict reload clears the legacy adapter and keeps named states", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap");
  await client.request("/console/scripts", post(A));
  f.cloudFiles.set("analysis", { id: "analysis", name: "analysis.py", code: "main_remote = 3", version: 8 });
  const before = await client.request<ScriptFile>(path("analysis"));
  const failures: ScriptSyncNotification[] = [];
  client.subscribeScriptSync!(value => failures.push(value));
  await client.request(path("analysis"), put("main_local = 2", before.version));
  await until(() => failures.some(value => value.sync_conflict));
  assert.equal((await client.request<ScriptFile>(path("analysis"))).code, "main_local = 2");
  const resolved = await client.request<ScriptFile>(path("analysis"), { headers: { "X-OpenEcon-Resolve-Conflict": "remote" } });
  assert.equal(resolved.code, "main_remote = 3"); assert.equal(resolved.version, 3);
  assert.ok(f.state?.scripts?.[A]);
  const saved = await client.request<ScriptFile>(path("analysis"), put("after_reload = 4", resolved.version));
  assert.equal(saved.pending_sync, true); assert.equal(saved.version, 4);
  await until(() => f.state?.base_code === "after_reload = 4");
  assert.equal(f.state?.cloud_version, 9);
});

test("reopening while already online flushes offline files without waiting for an online event", async () => {
  const f = fixture(), first = f.client(); await first.request("/bootstrap"); f.offline();
  await first.request("/console/scripts", post(A, "offline.py", "answer = 42"));
  first.cancelPending(); f.online(); const reopened = f.client();
  await reopened.request("/bootstrap");
  assert.equal(f.cloudFiles.get(A)?.code, "answer = 42");
  assert.equal(f.state?.scripts?.[A].cloud_version, 0);
  assert.equal((await reopened.request<ScriptFile>(path(A))).pending_sync, false);
});

test("named file size validation counts Unicode code points like the Python API", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap");
  await client.request("/console/scripts", post(A));
  const code = "# " + "😀".repeat(33000);
  const saved = await client.request<ScriptFile>(path(A), put(code, 0));
  assert.equal(saved.code, code); assert.equal(saved.pending_sync, true);
  await until(() => f.state?.scripts?.[A].base_code === code);
});

test("dirty-file autosave flush and cached target read finish while the cloud save is held", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(A, "a.py"); f.remoteFile(B, "b.py");
  await client.request("/bootstrap");
  const old = await client.request<ScriptFile>(path(A)); await client.request(path(B));
  const held = f.holdCloud(path(A));
  const writer = new VersionedScriptWriter(client, old.version, A);
  const saver = new LatestValueAutosaver({ initialValue: old.code!, save: (code) => writer.save(code) });
  saver.update("edited = 42");
  try {
    await saver.flush();
    await held.entered;
    let switched = false;
    const next = client.request<ScriptFile>(path(B)).then(file => { switched = true; return file; });
    await until(() => switched);
    assert.equal((await next).name, "b.py");
    assert.equal(f.localFiles.get(A)?.code, "edited = 42");
    assert.equal(f.cloudFiles.get(A)?.code, "remote = 1");
    assert.equal(saver.getSavedValue(), "edited = 42");
    assert.equal(f.cloudCalls.filter(call => call.path === path(B)).length, 1);
  } finally { saver.dispose(); held.release(); }
  await until(() => f.state?.scripts?.[A].base_code === "edited = 42");
});

test("main-file local CAS acknowledgement and reads bypass pending cloud synchronization", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap");
  const main = await client.request<ScriptFile>(path("analysis"));
  const held = f.holdCloud("/console/script");
  const pending = await client.request<ScriptFile>(path("analysis"), put("main = 42", main.version));
  await held.entered;
  try {
    assert.equal(pending.version, main.version + 1); assert.equal(pending.pending_sync, true);
    const read = await client.request<ScriptFile>(path("analysis"));
    assert.equal(read.code, "main = 42"); assert.equal(read.version, pending.version);
    assert.equal(f.state?.cloud_version, 7); assert.equal(f.cloudFiles.get("analysis")?.code, "main = 1");
    await assert.rejects(client.request(path("analysis"), put("stale", main.version)), isStatus(409));
  } finally { held.release(); }
  await until(() => f.state?.base_code === "main = 42");
  assert.equal((await client.request<ScriptFile>(path("analysis"))).pending_sync, false);
});

test("coalesced named writes preserve later edits when an older cloud acknowledgement arrives", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  const notifications: ScriptSyncNotification[] = [];
  client.subscribeScriptSync!(value => notifications.push(value));
  const held = f.holdCloud(path(A), "PUT", true);
  const first = await client.request<ScriptFile>(path(A), put("first = 1", 0)); await held.entered;
  const second = await client.request<ScriptFile>(path(A), put("second = 2", first.version));
  const third = await client.request<ScriptFile>(path(A), put("latest = 3", second.version));
  assert.equal(third.version, 3); assert.equal(f.cloudFiles.get(A)?.code, "first = 1");
  held.release();
  await until(() => notifications.some(value => value.file?.code === "latest = 3" && !value.file.pending_sync));
  assert.equal(f.localFiles.get(A)?.code, "latest = 3");
  const firstAck = notifications.find(value => value.file?.code === "latest = 3");
  assert.equal(firstAck?.file?.pending_sync, true, "late ack must not mark a newer edit synchronized");
  assert.deepEqual(f.cloudCalls.filter(call => call.path === path(A) && call.method === "PUT").map(call => call.body.code), ["first = 1", "latest = 3"]);
});

test("coalesced main writes use current cloud version without making local receipts stale", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap");
  const main = await client.request<ScriptFile>(path("analysis"));
  const held = f.holdCloud("/console/script", "PUT", true);
  const first = await client.request<ScriptFile>(path("analysis"), put("first = 1", main.version)); await held.entered;
  const latest = await client.request<ScriptFile>(path("analysis"), put("latest = 2", first.version));
  held.release(); await until(() => f.state?.base_code === "latest = 2");
  assert.equal(f.localFiles.get("analysis")?.version, latest.version);
  assert.deepEqual(f.cloudCalls.filter(call => call.path === "/console/script" && call.method === "PUT").map(call => call.body.version), [7, 8]);
  assert.equal(f.state?.cloud_version, 9);
});

test("overlapping local writers reject stale CAS before reaching cloud or replacing the draft", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  const held = f.holdCloud(path(A));
  const results = await Promise.allSettled([
    client.request(path(A), put("winner = 1", 0)), client.request(path(A), put("loser = 2", 0)),
  ]);
  try {
    assert.equal(results[0].status, "fulfilled"); assert.equal(results[1].status, "rejected");
    if (results[1].status === "rejected") assert.equal(results[1].reason.status, 409);
    assert.equal(f.localFiles.get(A)?.code, "winner = 1");
    await held.entered;
    assert.equal(f.cloudCalls.filter(call => call.path === path(A) && call.method === "PUT").length, 1);
  } finally { held.release(); }
  await until(() => f.state?.scripts?.[A].base_code === "winner = 1");
});

test("a disk failure leaves the old draft and schedules no cloud write", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  f.failLocal(path(A));
  await assert.rejects(client.request(path(A), put("not durable", 0)), /disk write failed/);
  assert.equal(f.localFiles.get(A)?.code, "remote = 1");
  assert.equal(f.cloudCalls.some(call => call.method === "PUT"), false);
});

test("background named authorization denial is durable, reported, and blocks cached access", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  const held = f.holdCloud(path(A));
  const pending = await client.request<ScriptFile>(path(A), put("preserved = 1", 0)); await held.entered;
  assert.equal(pending.pending_sync, true); f.deny(); held.release();
  await until(() => notices.some(value => value.sync_error?.status === 403));
  assert.equal(f.state?.access_denied, true); assert.equal(notices.at(-1)?.read_only, true);
  await assert.rejects(client.request(path(A)), isStatus(403));
  await assert.rejects(client.request(path(A), put("forbidden", pending.version)), isStatus(403));
  assert.equal(f.localFiles.get(A)?.code, "preserved = 1");
  client.cancelPending(); f.allow(); f.offline(); const reopened = f.client();
  await assert.rejects(reopened.request("/bootstrap"), isStatus(403));
});

test("inactive named conflicts persist and opening them never silently changes the local draft", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(A, "a.py"); f.remoteFile(B, "b.py");
  await client.request("/bootstrap"); await client.request(path(A)); await client.request(path(B));
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  f.remoteFile(A, "a.py", "peer = 2", 13);
  const pending = await client.request<ScriptFile>(path(A), put("mine = 1", 0));
  await until(() => notices.some(value => value.file?.id === A && value.sync_conflict));
  const reopened = await client.request<ScriptFile>(path(A));
  assert.equal(reopened.code, "mine = 1"); assert.equal(reopened.sync_conflict, true);
  await assert.rejects(client.request(path(A), put("must not replace", pending.version)), isStatus(409));
  await client.request(path(B), put("independent = 3", 0));
  await until(() => f.state?.scripts?.[B].base_code === "independent = 3");
  assert.equal(f.state?.scripts?.[A].conflict, true); assert.equal(f.cloudFiles.get(A)?.code, "peer = 2");
});

test("uncached files fetch once after the remote queue, then subsequent reads are local", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(A, "a.py"); f.remoteFile(B, "b.py");
  await client.request("/bootstrap"); await client.request(path(A));
  const held = f.holdCloud(path(A)); await client.request(path(A), put("edited = 1", 0)); await held.entered;
  let opened = false;
  const opening = client.request<ScriptFile>(path(B)).then(file => { opened = true; return file; });
  await new Promise(resolve => setTimeout(resolve, 5)); assert.equal(opened, false);
  held.release(); assert.equal((await opening).name, "b.py");
  const refreshing = f.holdCloud(path(B), "GET");
  const cached = await client.request<ScriptFile>(path(B)); await refreshing.entered;
  try {
    assert.equal(cached.code, "remote = 1");
    assert.equal((await client.request<ScriptFile>(path(B))).code, cached.code);
    assert.equal(f.cloudCalls.filter(call => call.path === path(B)).length, 2);
  } finally { refreshing.release(); client.cancelPending(); }
});

test("a delayed remote reload cannot overwrite a local edit made during its fetch", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  f.remoteFile(A, "remote.py", "new peer text", 13);
  const held = f.holdCloud(path(A), "GET");
  const reloading = client.request(path(A), remoteReload); await held.entered;
  const saved = await client.request<ScriptFile>(path(A), put("new local text", 0));
  assert.equal(saved.version, 1); held.release();
  await assert.rejects(reloading, isStatus(409));
  assert.equal(f.localFiles.get(A)?.code, "new local text");
  await until(() => Boolean(f.state?.scripts?.[A].conflict));
  assert.equal(f.cloudFiles.get(A)?.code, "new peer text");
});

test("closing after local receipt suppresses late cloud acknowledgements and notifications", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  const held = f.holdCloud(path(A), "PUT", true);
  await client.request(path(A), put("durable before close", 0)); await held.entered;
  const writes = f.localCalls.filter(call => call.method === "PUT").length;
  client.cancelPending(); held.release(); await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(notices.length, 0);
  assert.equal(f.localCalls.filter(call => call.method === "PUT").length, writes);
  assert.equal(f.localFiles.get(A)?.code, "durable before close");
  await assert.rejects(client.request(path(A)), { name: "AbortError" });
  const reopened = f.client(); await reopened.request("/bootstrap");
  assert.equal((await reopened.request<ScriptFile>(path(A))).code, "durable before close");
  assert.equal(f.state?.scripts?.[A].base_code, "durable before close");
});

test("unchanged explicit remote reload leaves the active local writer version valid", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap");
  const file = await client.request<ScriptFile>(path(A));
  const writer = new VersionedScriptWriter(client, file.version, A);
  const unchanged = await client.request<ScriptFile>(path(A), remoteReload);
  assert.equal(unchanged.version, file.version);
  const pending = await writer.save("writer remains valid"); assert.equal(pending.pending_sync, true);
  await until(() => f.state?.scripts?.[A].base_code === "writer remains valid");
});

test("cached named opening refreshes peers in the background and reports an actual source change", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap");
  const initial = await client.request<ScriptFile>(path(A));
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  f.remoteFile(A, "remote.py", "peer updated text", 13);
  const held = f.holdCloud(path(A), "GET");
  const cached = await client.request<ScriptFile>(path(A)); await held.entered;
  assert.equal(cached.code, initial.code); assert.equal(cached.version, initial.version);
  assert.equal(notices.length, 0); held.release();
  await until(() => notices.some(value => value.source_changed));
  const changed = notices.find(value => value.source_changed)!;
  assert.equal(changed.file?.code, "peer updated text"); assert.equal(changed.file?.version, initial.version + 1);
  assert.equal(changed.file?.pending_sync, false);
  assert.equal(f.localFiles.get(A)?.code, "peer updated text");
});

test("unchanged background refresh performs no cache write and keeps the active writer's local CAS", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap");
  const initial = await client.request<ScriptFile>(path(A));
  const before = f.localCalls.filter(call => call.path.startsWith("/desktop-scripts/")).length;
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  const held = f.holdCloud(path(A), "GET");
  await client.request(path(A)); await held.entered; held.release();
  await until(() => notices.length > 0);
  assert.equal(notices.some(value => value.source_changed), false);
  assert.equal(f.localCalls.filter(call => call.path.startsWith("/desktop-scripts/")).length, before);
  assert.equal(f.localFiles.get(A)?.version, initial.version);
  const writer = new VersionedScriptWriter(client, initial.version, A);
  await writer.save("same writer valid"); await until(() => f.state?.scripts?.[A].base_code === "same writer valid");
});

test("local edits during a cached refresh remain durable when a peer changed the remote source", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  f.remoteFile(A, "remote.py", "peer = 2", 13);
  const held = f.holdCloud(path(A), "GET"); await client.request(path(A)); await held.entered;
  const pending = await client.request<ScriptFile>(path(A), put("late local = 3", 0));
  assert.equal(pending.pending_sync, true); held.release();
  await until(() => notices.some(value => value.sync_conflict));
  assert.equal(f.localFiles.get(A)?.code, "late local = 3");
  assert.equal(f.cloudFiles.get(A)?.code, "peer = 2");
  assert.equal(notices.some(value => value.source_changed), false);
  assert.equal(f.state?.scripts?.[A].conflict, true);
});

test("a local edit during an unchanged refresh stays pending until the queued upload acknowledges it", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  const held = f.holdCloud(path(A), "GET"); await client.request(path(A)); await held.entered;
  await client.request(path(A), put("late local = 3", 0)); held.release();
  await until(() => notices.some(value => value.file?.code === "late local = 3" && !value.file.pending_sync));
  assert.equal(f.state?.scripts?.[A].conflict, undefined);
  assert.equal(notices[0].file?.code, "late local = 3"); assert.equal(notices[0].file?.pending_sync, true);
  assert.equal(notices.some(value => value.source_changed), false);
});

test("cached main opening reports a peer source change without waiting or rewriting the caller's snapshot", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap");
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  f.cloudFiles.set("analysis", { id: "analysis", name: "analysis.py", code: "peer main = 2", version: 8 });
  const held = f.holdCloud("/console/script", "GET");
  const initial = await client.request<ScriptFile>(path("analysis")); await held.entered;
  assert.equal(initial.code, "main = 1"); held.release();
  await until(() => notices.some(value => value.source_changed));
  assert.equal(initial.code, "main = 1", "returned snapshot must remain immutable");
  assert.equal(f.localFiles.get("analysis")?.code, "peer main = 2");
  assert.equal(notices.find(value => value.source_changed)?.file?.version, initial.version + 1);
  assert.equal(f.state?.cloud_version, 8);
});

test("main edits during a background refresh preserve local and peer source and expose conflict", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap");
  f.cloudFiles.set("analysis", { id: "analysis", name: "analysis.py", code: "peer main = 2", version: 8 });
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  const held = f.holdCloud("/console/script", "GET");
  const initial = await client.request<ScriptFile>(path("analysis")); await held.entered;
  await client.request(path("analysis"), put("mine main = 3", initial.version)); held.release();
  await until(() => notices.some(value => value.sync_conflict));
  assert.equal(f.localFiles.get("analysis")?.code, "mine main = 3");
  assert.equal(f.cloudFiles.get("analysis")?.code, "peer main = 2");
  assert.equal(notices.some(value => value.source_changed), false);
  client.cancelPending(); const reopened = f.client(); await reopened.request("/bootstrap");
  assert.equal((await reopened.request<ScriptFile>(path("analysis"))).sync_conflict, true);
});

test("authorization revoked during a background cached read locks every cached file", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  const held = f.holdCloud(path(A), "GET"); const cached = await client.request<ScriptFile>(path(A)); await held.entered;
  assert.equal(cached.code, "remote = 1"); f.deny(); held.release();
  await until(() => notices.some(value => value.sync_error?.status === 403));
  assert.equal(f.state?.access_denied, true);
  assert.equal(notices.at(-1)?.read_only, true);
  await assert.rejects(client.request(path("analysis")), isStatus(403));
  await assert.rejects(client.request(path(A)), isStatus(403));
});

test("background refresh failures are reported while cached source remains available", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  const notices: ScriptSyncNotification[] = []; client.subscribeScriptSync!(value => notices.push(value));
  f.oldServer(); assert.equal((await client.request<ScriptFile>(path(A))).code, "remote = 1");
  await until(() => notices.some(value => value.sync_error?.status === 404));
  assert.equal(f.state?.access_denied, undefined); assert.equal(f.localFiles.get(A)?.code, "remote = 1");
});

test("a queued named acknowledgement merges the newest legacy draft, role and other file metadata", async () => {
  let state: DesktopSyncState = { cloud_version: 7, base_code: "legacy old", role: "editor",
    scripts: { [A]: { name: "a.py", cloud_version: 1, base_code: "old", local_version: 0 } } };
  let file: ScriptFile = { id: A, name: "a.py", code: "old", version: 0 };
  let entered!: () => void, release!: () => void;
  const persisting = new Promise<void>(resolve => { entered = resolve; });
  const wait = new Promise<void>(resolve => { release = resolve; });
  const local: WorkspaceClient = { projectId: null, teams: false, cancelPending() {},
    connect: async () => ({ token: "local", version: "test", environment: "local" }),
    async request<T>(pathname: string, options: RequestInit = {}) {
      if (pathname.startsWith("/desktop-scripts/")) {
        const body = JSON.parse(String(options.body));
        assert.equal(body.version, file.version);
        file = { ...file, code: body.code, version: file.version + 1 };
      }
      return structuredClone(file) as T;
    },
  };
  const sync = createNamedScriptSync({ local, state: () => state,
    cloud: async <T>() => ({ id: A, name: "a.py", code: "remote updated", version: 2 }) as T,
    persist: async update => {
      entered(); await wait;
      state = typeof update === "function" ? update(state) : update;
    },
  });
  const refreshing = sync.read(A); await persisting;
  // A legacy source save and a layout/auth operation can have updated unrelated
  // fields since the named response started. Its ack must preserve all of them.
  state = { ...state, cloud_version: 8, base_code: "legacy newest", role: "viewer",
    scripts: { ...state.scripts, [B]: { name: "b.py", cloud_version: 4, local_version: 3, base_code: "other newest", conflict: true } },
    script_catalog: [{ id: "analysis", name: "analysis.py", version: 8 }] };
  release(); await refreshing;
  assert.equal(state.cloud_version, 8); assert.equal(state.base_code, "legacy newest"); assert.equal(state.role, "viewer");
  assert.equal(state.scripts?.[B].conflict, true); assert.equal(state.scripts?.[B].base_code, "other newest");
  assert.equal(state.script_catalog?.[0].version, 8);
  assert.equal(state.scripts?.[A].base_code, "remote updated"); assert.equal(state.scripts?.[A].cloud_version, 2);
});

test("a lost main upload acknowledgement reconciles exact cloud content without bumping local CAS", async () => {
  const f = fixture(), client = f.client(); await client.request("/bootstrap");
  const initial = await client.request<ScriptFile>(path("analysis"));
  f.loseAck("PUT");
  const first = await client.request<ScriptFile>(path("analysis"), put("lost ack = 1", initial.version));
  await until(() => f.cloudFiles.get("analysis")?.code === "lost ack = 1");
  // A later cached open performs the same exact-content reconciliation in its
  // background refresh, leaving the local writer's CAS version unchanged.
  await client.request(path("analysis")); await until(() => f.state?.base_code === "lost ack = 1");
  assert.equal(f.localFiles.get("analysis")?.version, first.version);
  const second = await client.request<ScriptFile>(path("analysis"), put("next = 2", first.version));
  await until(() => f.state?.base_code === "next = 2");
  assert.equal(f.localFiles.get("analysis")?.version, second.version);
  assert.equal(f.state?.cloud_version, 9);
});

test("known authorization denial blocks access immediately while its durable metadata write is held", async () => {
  const f = fixture(), client = f.client(); f.remoteFile(); await client.request("/bootstrap"); await client.request(path(A));
  const cloud = f.holdCloud(path(A), "GET"); await client.request(path(A)); await cloud.entered;
  const metadata = f.holdState(); f.deny(); cloud.release(); await metadata.entered;
  try {
    assert.equal(f.state?.access_denied, undefined, "durable denial write is deliberately not acknowledged yet");
    await assert.rejects(client.request(path(A)), isStatus(403));
    await assert.rejects(client.request(path("analysis")), isStatus(403));
    await assert.rejects(client.request(path(A), put("forbidden during metadata write", 0)), isStatus(403));
    assert.equal(f.localFiles.get(A)?.code, "remote = 1");
  } finally { metadata.release(); }
  await until(() => Boolean(f.state?.access_denied));
  f.allow(); await client.request("/bootstrap");
  assert.equal(f.state?.access_denied, false);
  assert.equal((await client.request<ScriptFile>(path(A))).code, "remote = 1");
});

test("Markdown and LaTeX source cache and offline saves use the same local CAS and cloud acknowledgements", async () => {
  for (const [id, name, content] of [[A, "notes.md", "# Results"], [B, "table.tex", "\\section{Results}"]]) {
    const f = fixture(), client = f.client();
    f.remoteFile(id, name, content, 12);
    try {
      await client.request("/bootstrap");
      const opened = await client.request<ScriptFile>(path(id));
      assert.equal(opened.name, name);
      f.offline();
      const draft = content + "\nSaved locally";
      const receipt = await client.request<ScriptFile>(path(id), put(draft, opened.version));
      assert.equal(receipt.pending_sync, true);
      assert.equal(f.localFiles.get(id)?.code, draft);
      assert.equal((await client.request<ScriptFile>(path(id))).name, name);
      f.online();
      await client.request("/bootstrap");
      await until(() => f.cloudFiles.get(id)?.code === draft);
      assert.equal(f.cloudFiles.get(id)?.name, name);
      assert.equal([...f.localCalls, ...f.cloudCalls].some(c => c.path.includes("execute")), false);
    } finally { client.cancelPending(); }
  }
});

test("legacy .py-only cloud creation keeps Markdown and LaTeX durable and pending until an upgraded retry", async () => {
  for (const name of ["notes.md", "table.tex"]) {
    const f = fixture(), client = f.client();
    f.legacyDocuments();
    try {
      await client.request("/bootstrap");
      const created = await client.request<ScriptFile>("/console/scripts", post(A, name, ""));
      assert.equal(created.name, name);
      assert.equal(created.pending_sync, true);
      assert.equal(f.state?.scripts?.[A].cloud_version, null);
      assert.equal(f.state?.scripts?.[A].conflict, undefined);
      assert.equal(f.cloudFiles.has(A), false);
      const saved = await client.request<ScriptFile>(path(A), put("owned document content", created.version));
      assert.equal(saved.pending_sync, true);
      assert.equal(f.localFiles.get(A)?.code, "owned document content");
      const reopened = f.client();
      await reopened.request("/bootstrap");
      const read = await reopened.request<ScriptFile>(path(A));
      assert.equal(read.name, name);
      assert.equal(read.code, "owned document content");
      assert.equal(read.pending_sync, true);
      f.documentUpgrade();
      await reopened.request("/bootstrap");
      await until(() => f.cloudFiles.get(A)?.code === "owned document content");
      assert.equal((await reopened.request<ScriptFile>(path(A))).pending_sync, false);
      const posts = f.cloudCalls.filter(call => call.path === "/console/scripts" && call.method === "POST");
      assert.ok(posts.length >= 2);
      assert.ok(posts.every(call => call.body.id === A && call.body.name === name && call.body.code === ""));
      reopened.cancelPending();
    } finally { client.cancelPending(); }
  }
});

test("legacy document compatibility never downgrades authorization, conflicts, other validation or Python errors", async () => {
  for (const [name, error] of [
    ["notes.md", new ApiError("Güvenli, kısa bir .py dosya adı kullanın.", 401, "INVALID_SCRIPT")],
    ["notes.md", new ApiError("Güvenli, kısa bir .py dosya adı kullanın.", 403, "INVALID_SCRIPT")],
    ["notes.md", new ApiError("different validation", 422, "INVALID_SCRIPT")],
    ["notes.md", new ApiError("Güvenli, kısa bir .py dosya adı kullanın.", 422, "OTHER_VALIDATION")],
    ["notes.md", new ApiError("Güvenli, kısa bir .py dosya adı kullanın.", 409, "VERSION_CONFLICT")],
    ["code.py", new ApiError("Güvenli, kısa bir .py dosya adı kullanın.", 422, "INVALID_SCRIPT")],
  ] as const) {
    const f = fixture(), client = f.client();
    try {
      await client.request("/bootstrap");
      f.rejectCreation(error);
      await assert.rejects(client.request("/console/scripts", post(A, name, "")), error.status === 409 ? isStatus(404) : failure => failure === error);
      assert.equal(f.state?.scripts?.[A].cloud_version, null);
      assert.equal(f.cloudFiles.has(A), false);
      assert.equal(f.localFiles.get(A)?.name, name, "a non-acknowledged creation never discards its durable source");
      if ([401, 403].includes(error.status)) assert.equal(f.state?.access_denied, true);
    } finally { client.cancelPending(); }
  }
});

test("conflict comparison preserves all sources and explicitly commits a merged draft with cloud CAS", async () => {
  for (const id of ["analysis", A]) {
    const f = fixture();
    const client = f.client();
    await client.request("/bootstrap");
    if (id === A) {
      f.cloudFiles.set(A, {
        id: A,
        name: "file.py",
        code: "base = 1",
        version: 3,
      });
      await client.request(path(A));
    }
    const base = f.localFiles.get(id)!;
    f.localFiles.set(id, {
      ...base,
      code: "local = 2",
      version: base.version + 1,
    });
    const remote = f.cloudFiles.get(id)!;
    f.cloudFiles.set(id, {
      ...remote,
      code: "cloud = 3",
      version: remote.version + 1,
    });
    const before = structuredClone([...f.localFiles]);
    const snapshot = await client.readScriptConflict!(id);
    assert.deepEqual(
      [...f.localFiles],
      before,
      "reading comparisons cannot replace drafts",
    );
    assert.equal(snapshot.local.code, "local = 2");
    assert.equal(snapshot.remote.code, "cloud = 3");
    assert.equal(snapshot.base, id === A ? "base = 1" : "main = 1");
    const saved = await client.resolveScriptConflict!(
      snapshot,
      "local = 2\ncloud = 3",
    );
    assert.equal(saved.code, "local = 2\ncloud = 3");
    assert.equal(saved.pending_sync, false);
    assert.equal(f.cloudFiles.get(id)!.code, saved.code);
    assert.equal(f.localFiles.get(id)!.code, saved.code);
    const put = f.cloudCalls.findLast((item) => item.method === "PUT")!;
    assert.equal(put.body.version, snapshot.remote.version);
    client.cancelPending();
  }
});

test("a second cloud writer keeps the exact local source and rejected comparison intact", async () => {
  const f = fixture();
  const client = f.client();
  await client.request("/bootstrap");
  const local = f.localFiles.get("analysis")!;
  f.localFiles.set("analysis", {
    ...local,
    code: "local",
    version: local.version + 1,
  });
  const snapshot = await client.readScriptConflict!("analysis");
  const remote = f.cloudFiles.get("analysis")!;
  f.cloudFiles.set("analysis", {
    ...remote,
    code: "second writer",
    version: remote.version + 1,
  });
  await assert.rejects(
    client.resolveScriptConflict!(snapshot, "merged"),
    isStatus(409),
  );
  assert.equal(f.localFiles.get("analysis")!.code, "local");
  assert.equal(f.cloudFiles.get("analysis")!.code, "second writer");
  assert.equal(snapshot.remote.code, "main = 1");
  client.cancelPending();
});

test("a second local writer prevents cloud mutation and lost cloud acknowledgement preserves both drafts", async () => {
  const f = fixture();
  const client = f.client();
  await client.request("/bootstrap");
  const snapshot = await client.readScriptConflict!("analysis");
  f.localFiles.set("analysis", {
    ...snapshot.local,
    code: "other local",
    version: snapshot.local.version + 1,
  });
  await assert.rejects(
    client.resolveScriptConflict!(snapshot, "merged"),
    isStatus(409),
  );
  assert.equal(f.cloudFiles.get("analysis")!.code, snapshot.remote.code);
  const next = await client.readScriptConflict!("analysis");
  f.loseAck("PUT");
  await assert.rejects(
    client.resolveScriptConflict!(next, "merged"),
    TypeError,
  );
  assert.equal(f.localFiles.get("analysis")!.code, "other local");
  assert.equal(f.cloudFiles.get("analysis")!.code, "merged");
  const retry = await client.readScriptConflict!("analysis");
  await client.resolveScriptConflict!(retry, "merged");
  assert.equal(f.localFiles.get("analysis")!.code, "merged");
  client.cancelPending();
});

test("comparison authorization denial is durable and can never authorize a merge", async () => {
  const f = fixture();
  const client = f.client();
  await client.request("/bootstrap");
  const snapshot = await client.readScriptConflict!("analysis");
  const before = structuredClone([...f.localFiles]);
  f.deny();
  await assert.rejects(client.readScriptConflict!("analysis"), isStatus(403));
  assert.equal(f.state?.access_denied, true);
  await assert.rejects(
    client.resolveScriptConflict!(snapshot, "merged"),
    isStatus(403),
  );
  assert.deepEqual([...f.localFiles], before);
  client.cancelPending();
});
