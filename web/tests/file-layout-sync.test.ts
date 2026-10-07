import { test } from "node:test";
import assert from "node:assert/strict";
import { ApiError, type WorkspaceClient } from "../src/api.ts";
import {
  createFileLayoutSync,
  type FileLayoutSyncCache,
} from "../src/file-layout-sync.ts";
import type { DesktopSyncState } from "../src/desktop-script-sync.ts";
import type { FileLayout, FileLayoutEntry } from "../src/file-layout.ts";
import { createHybridWorkspace } from "../src/desktop-sync.ts";

const DATA = "11111111-1111-4111-8111-111111111111";
const CLOUD = "a".repeat(32);
const FOLDER = "f".repeat(32);
const main: FileLayoutEntry = {
  kind: "script",
  id: "analysis",
  name: "analysis.py",
  parent: null,
};
const dataset: FileLayoutEntry = {
  kind: "dataset",
  id: DATA,
  name: "wages.csv",
  parent: null,
};
const folder: FileLayoutEntry = {
  kind: "folder",
  id: FOLDER,
  name: "Models",
  parent: null,
};
const put = (document: FileLayout) => JSON.stringify(document);
const isStatus = (status: number) => (error: unknown) =>
  error instanceof ApiError && error.status === status;

function fixture() {
  let local: FileLayout = { version: 0, entries: [main, dataset] };
  let remote: FileLayout = {
    version: 4,
    entries: [main, { ...dataset, id: CLOUD }],
  };
  let cache: FileLayoutSyncCache | null = null;
  let state: DesktopSyncState | null = {
    role: "editor",
    base_code: "",
    cloud_version: 1,
  };
  let online = true,
    supported = true,
    forbidden = false,
    loseAck = false;
  let rejectInventory = false;
  let legacyDocumentNames = false;
  let mappings = [{ cloud_id: CLOUD, dataset_id: DATA }];
  let localOnly: string[] = [];
  let held: {
    method: string;
    entered: () => void;
    wait: Promise<void>;
  } | null = null;
  const calls: {
    target: "local" | "cloud";
    path: string;
    method: string;
    body: any;
  }[] = [];
  const localClient: WorkspaceClient = {
    projectId: null,
    teams: false,
    cancelPending() {},
    async connect() {
      return { token: "local", version: "test", environment: "local" };
    },
    async request<T>(path: string, options: RequestInit = {}): Promise<T> {
      const method = options.method ?? "GET";
      const body = options.body ? JSON.parse(String(options.body)) : undefined;
      calls.push({ target: "local", path, method, body });
      if (path === "/desktop/file-layout") {
        if (method === "PUT") {
          if (body.version !== local.version)
            throw new ApiError("local changed", 409, "VERSION_CONFLICT");
          local = { version: local.version + 1, entries: body.entries };
        }
        return structuredClone(local) as T;
      }
      if (path === "/desktop/file-layout-sync") {
        if (method === "PUT") {
          if (body.version !== (cache?.version ?? 0))
            throw new ApiError("cache changed", 409, "VERSION_CONFLICT");
          cache = { ...body, version: body.version + 1 };
        }
        return structuredClone(cache) as T;
      }
      if (path === "/desktop-cached-files")
        return structuredClone({ files: mappings, local_only: localOnly }) as T;
      throw new Error(`unexpected local path ${path}`);
    },
  };
  async function cloud<T>(path: string, options: RequestInit = {}) {
    const method = options.method ?? "GET";
    const body = options.body ? JSON.parse(String(options.body)) : undefined;
    calls.push({ target: "cloud", path, method, body });
    if (forbidden) throw new ApiError("removed", 403, "ROLE_REQUIRED");
    if (!online) throw new TypeError("offline");
    if (!supported) throw new ApiError("old deployment", 404, "NOT_FOUND");
    assert.equal(path, "/files/layout");
    const captured = structuredClone(remote);
    if (held?.method === method) {
      const pending = held;
      held = null;
      pending.entered();
      await pending.wait;
    }
    if (method === "PUT") {
      if (legacyDocumentNames && body.entries.some((entry: FileLayoutEntry) => entry.kind === "script" && /\.(md|tex)$/.test(entry.name)))
        throw new ApiError("Python dosyası için bir .py adı kullanın.", 422, "INVALID_FILE_LAYOUT");
      if (rejectInventory)
        throw new ApiError("inventory changed", 409, "FILE_INVENTORY_CONFLICT");
      if (body.version !== remote.version)
        throw new ApiError("peer changed", 409, "VERSION_CONFLICT");
      remote = { version: remote.version + 1, entries: body.entries };
      if (loseAck) {
        loseAck = false;
        throw new TypeError("lost acknowledgement");
      }
    }
    return (method === "GET" ? captured : structuredClone(remote)) as T;
  }
  function client() {
    return createFileLayoutSync({
      local: localClient,
      state: () => state,
      persist: async (next) => {
        state = next;
      },
      cloud,
    });
  }
  return {
    client,
    localClient,
    cloud,
    calls,
    get local() {
      return local;
    },
    get remote() {
      return remote;
    },
    get cache() {
      return cache;
    },
    get state() {
      return state;
    },
    offline() {
      online = false;
    },
    online() {
      online = true;
    },
    oldServer() {
      supported = false;
    },
    deny() {
      forbidden = true;
    },
    inventoryConflict() {
      rejectInventory = true;
    },
    legacyDocuments() { legacyDocumentNames = true; },
    documentUpgrade() { legacyDocumentNames = false; },
    addLocalScript(entry: FileLayoutEntry) {
      local = { ...local, entries: [...local.entries, entry] };
    },
    shareScript(entry: FileLayoutEntry) {
      remote = { ...remote, entries: [...remote.entries, entry] };
    },
    viewer() {
      state = { ...state!, role: "viewer" };
    },
    loseAck() {
      loseAck = true;
    },
    peer(entries: FileLayoutEntry[]) {
      remote = { version: remote.version + 1, entries };
    },
    freshRemote() {
      remote = { ...remote, version: 0 };
    },
    unmap() {
      mappings = [];
    },
    delayCloud(method: string) {
      let entered!: () => void, release!: () => void;
      const started = new Promise<void>((resolve) => {
        entered = resolve;
      });
      const wait = new Promise<void>((resolve) => {
        release = resolve;
      });
      held = { method, entered, wait };
      return { started, release };
    },
    addLocalDataset() {
      local = {
        ...local,
        entries: [
          ...local.entries,
          {
            ...dataset,
            id: "22222222-2222-4222-8222-222222222222",
            name: "local.csv",
          },
        ],
      };
    },
    addPrivateDataset(entry: FileLayoutEntry) {
      localOnly.push(entry.id);
      local = { ...local, entries: [...local.entries, entry] };
    },
  };
}

function hybridFixture() {
  const f = fixture();
  let persisted: DesktopSyncState = {
    role: "editor",
    base_code: "original = 1",
    cloud_version: 1,
  };
  let code = "original = 1";
  let remoteCode = code;
  let remoteVersion = 1;
  let draftWait: { started: () => void; wait: Promise<void> } | null = null;
  const cloudCalls: { path: string; method: string; body: any }[] = [];
  const localCalls: { path: string; method: string }[] = [];
  const catalog = {
    scripts: [{ id: "analysis", name: "analysis.py", version: 1 }],
  };
  const named = new Map<
    string,
    { id: string; name: string; code: string; version: number }
  >();
  const local: WorkspaceClient = {
    ...f.localClient,
    async request<T>(path: string, options: RequestInit = {}) {
      localCalls.push({ path, method: options.method ?? "GET" });
      if (path === "/desktop-sync-state") {
        if (options.method === "PUT")
          persisted = JSON.parse(String(options.body));
        return structuredClone(persisted) as T;
      }
      if (path === "/console/scripts") return structuredClone(catalog) as T;
      if (named.has(path.split("/").at(-1)!))
        return structuredClone(named.get(path.split("/").at(-1)!)) as T;
      if (path === "/console/script") {
        if (options.method === "PUT")
          code = JSON.parse(String(options.body)).code;
        return { code, name: "analysis.py", version: 1 } as T;
      }
      return f.localClient.request<T>(path, options);
    },
  };
  const hybrid = createHybridWorkspace("b".repeat(32), "editor", {
    actorUid: "editor",
    local,
    cacheFile: async () => {
      throw new Error("No downloads expected");
    },
    async cloud<T>(path: string, options: RequestInit = {}) {
      const body = options.body ? JSON.parse(String(options.body)) : undefined;
      cloudCalls.push({ path, method: options.method ?? "GET", body });
      if (path === "/console/scripts" && options.method === "POST") {
        f.shareScript({
          kind: "script",
          id: body.id,
          name: body.name,
          parent: null,
        });
        return { ...body, version: 1 } as T;
      }
      if (path === "/console/scripts") return structuredClone(catalog) as T;
      if (path === "/console/script") {
        if (draftWait) {
          const pending = draftWait;
          draftWait = null;
          pending.started();
          await pending.wait;
        }
        assert.equal(body.version, remoteVersion);
        remoteCode = body.code;
        remoteVersion++;
        return {
          name: "analysis.py",
          code: remoteCode,
          version: remoteVersion,
        } as T;
      }
      return f.cloud<T>(path, options);
    },
  });
  return {
    ...f,
    hybrid,
    cloudCalls,
    localCalls,
    get code() {
      return code;
    },
    get state() {
      return persisted;
    },
    pendingScript(id: string) {
      const document = {
        id,
        name: "pending.py",
        code: "unsaved = (",
        version: 1,
      };
      named.set(id, document);
      catalog.scripts.push({ id, name: document.name, version: 1 });
      persisted.scripts = {
        ...persisted.scripts,
        [id]: {
          name: document.name,
          cloud_version: null,
          local_version: 1,
          base_code: document.code,
        },
      };
      f.addLocalScript({
        kind: "script",
        id,
        name: document.name,
        parent: null,
      });
      return document;
    },
    delayDraft() {
      let started!: () => void, release!: () => void;
      const entered = new Promise<void>((resolve) => {
        started = resolve;
      });
      draftWait = {
        started,
        wait: new Promise<void>((resolve) => {
          release = resolve;
        }),
      };
      return { started: entered, release };
    },
  };
}

async function waitUntil(predicate: () => boolean) {
  const deadline = Date.now() + 1000;
  while (!predicate()) {
    if (Date.now() > deadline)
      assert.fail("Expected background synchronization did not finish");
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

test("cloud layout maps stable data IDs without changing canonical files and stays metadata only", async () => {
  const f = fixture(),
    sync = f.client();
  f.peer([
    folder,
    { ...main, name: "model.py", parent: FOLDER },
    { ...dataset, id: CLOUD, name: "survey.csv", parent: FOLDER },
  ]);
  const first = await sync.read();
  assert.equal(first.entries[2].id, DATA);
  assert.equal(first.entries[2].name, "survey.csv");
  assert.equal(first.pending_sync, false);
  assert.equal(f.cache?.base_entries[2].id, CLOUD);
  assert.equal(
    f.calls.some((call) =>
      /execute|console\/script|datasets\/upload/.test(call.path),
    ),
    false,
  );
  assert.equal(
    f.calls.some((call) =>
      String(JSON.stringify(call.body)).includes('"code"'),
    ),
    false,
  );
  assert.deepEqual((await f.client().read()).entries, first.entries);
});

test("hybrid acknowledges durable local metadata before its serialized cloud synchronization", async () => {
  const f = fixture();
  let persisted: DesktopSyncState = {
    role: "editor",
    base_code: "",
    cloud_version: 1,
  };
  const scripts = {
    scripts: [{ id: "analysis", name: "analysis.py", version: 1 }],
  };
  const cloudPaths: string[] = [];
  const local: WorkspaceClient = {
    ...f.localClient,
    async request<T>(path: string, options: RequestInit = {}) {
      if (path === "/desktop-sync-state") {
        if (options.method === "PUT")
          persisted = JSON.parse(String(options.body));
        return structuredClone(persisted) as T;
      }
      if (path === "/console/scripts") return structuredClone(scripts) as T;
      return f.localClient.request<T>(path, options);
    },
  };
  const hybrid = createHybridWorkspace("b".repeat(32), "editor", {
    actorUid: "editor",
    local,
    cacheFile: async () => {
      throw new Error("No dataset downloads expected");
    },
    async cloud<T>(path: string, options?: RequestInit) {
      cloudPaths.push(path);
      return path === "/console/scripts"
        ? (structuredClone(scripts) as T)
        : f.cloud<T>(path, options);
    },
  });
  const loaded = await hybrid.request<FileLayout>("/files/layout");
  const saved = await hybrid.request<FileLayout>("/files/layout", {
    method: "PUT",
    body: put({ version: loaded.version, entries: [dataset, main] }),
  });
  assert.equal(saved.pending_sync, true);
  const settled = await hybrid.request<FileLayout>("/files/layout");
  assert.equal(settled.pending_sync, false);
  assert.equal(persisted.script_catalog?.[0].id, "analysis");
  assert.deepEqual(cloudPaths.slice(0, 2), [
    "/console/scripts",
    "/files/layout",
  ]);
  await assert.rejects(
    hybrid.request("/files/layout", { method: "DELETE" }),
    isStatus(405),
  );
  assert.equal(
    cloudPaths.some((path) => path.includes("execute")),
    false,
  );
  hybrid.cancelPending();
});

test("folder, order, and logical renames persist first locally then cloud CAS", async () => {
  const f = fixture(),
    sync = f.client(),
    initial = await sync.read();
  const desired = [
    folder,
    { ...dataset, parent: FOLDER },
    { ...main, name: "regression.py" },
  ];
  const saved = await sync.save(
    put({ version: initial.version, entries: desired }),
  );
  assert.deepEqual(saved.entries, desired);
  assert.equal(saved.pending_sync, false);
  assert.equal(f.remote.entries[1].id, CLOUD);
  const localPut = f.calls.findIndex(
    (call) => call.path === "/desktop/file-layout" && call.method === "PUT",
  );
  const cloudPut = f.calls.findIndex(
    (call) => call.target === "cloud" && call.method === "PUT",
  );
  assert.ok(localPut < cloudPut);
  assert.equal(f.calls[cloudPut].body.version, 4);
  assert.equal(f.cache?.cloud_version, 5);
});

test("offline ordering survives reopen and retry without overwriting a cloud base", async () => {
  const f = fixture(),
    sync = f.client(),
    initial = await sync.read();
  f.offline();
  const desired = [{ ...dataset, name: "sample.csv" }, main];
  const saved = await sync.save(
    put({ version: initial.version, entries: desired }),
  );
  assert.equal(saved.pending_sync, true);
  const reopened = f.client();
  assert.deepEqual((await reopened.read()).entries, desired);
  f.online();
  await reopened.retry();
  assert.equal((await reopened.read()).pending_sync, false);
  assert.deepEqual(f.remote.entries, [
    { ...dataset, id: CLOUD, name: "sample.csv" },
    main,
  ]);
});

test("a teammate reorder creates durable explicit conflict and keeps both layouts", async () => {
  const f = fixture(),
    sync = f.client(),
    initial = await sync.read();
  f.offline();
  const local = [dataset, { ...main, name: "mine.py" }];
  await sync.save(put({ version: initial.version, entries: local }));
  const peer = [
    folder,
    { ...main, name: "peer.py", parent: FOLDER },
    { ...dataset, id: CLOUD },
  ];
  f.peer(peer);
  f.online();
  const reopened = f.client(),
    pending = await reopened.read();
  assert.equal(pending.sync_conflict, true);
  assert.deepEqual(pending.entries, local);
  assert.deepEqual(f.remote.entries, peer);
  assert.equal(
    f.calls.filter((call) => call.target === "cloud" && call.method === "PUT")
      .length,
    0,
  );
  await assert.rejects(
    reopened.save(put({ version: pending.version, entries: local })),
    isStatus(409),
  );
  assert.equal(f.cache?.conflict, true);
});

test("resolving a conflict fetches the current remote layout; it never silently merges", async () => {
  const f = fixture(),
    sync = f.client(),
    initial = await sync.read();
  f.offline();
  await sync.save(put({ version: initial.version, entries: [dataset, main] }));
  f.peer([folder, { ...main, parent: FOLDER }, { ...dataset, id: CLOUD }]);
  f.online();
  await sync.read();
  const accepted = await sync.read(true);
  assert.equal(accepted.sync_conflict, false);
  assert.equal(accepted.pending_sync, false);
  assert.equal(accepted.entries[1].parent, FOLDER);
  assert.equal(accepted.entries[2].id, DATA);
  assert.equal(
    f.calls.filter((call) => call.target === "cloud" && call.method === "PUT")
      .length,
    0,
  );
});

test("a lost successful write acknowledgement settles by exact readback on reopen", async () => {
  const f = fixture(),
    sync = f.client(),
    initial = await sync.read();
  f.loseAck();
  const saved = await sync.save(
    put({ version: initial.version, entries: [dataset, main] }),
  );
  assert.equal(saved.pending_sync, true);
  const puts = f.calls.filter(
    (call) => call.target === "cloud" && call.method === "PUT",
  ).length;
  const reopened = await f.client().read();
  assert.equal(reopened.pending_sync, false);
  assert.equal(reopened.sync_conflict, false);
  assert.equal(
    f.calls.filter((call) => call.target === "cloud" && call.method === "PUT")
      .length,
    puts,
  );
});

test("a peer update between read and save is never replaced by stale CAS", async () => {
  const f = fixture(),
    sync = f.client(),
    initial = await sync.read();
  const peer = [
    { ...main, name: "peer.py" },
    { ...dataset, id: CLOUD },
  ];
  f.peer(peer);
  await assert.rejects(
    sync.save(put({ version: initial.version, entries: [dataset, main] })),
    isStatus(409),
  );
  assert.deepEqual(f.remote.entries, peer);
  assert.deepEqual(f.local.entries, [dataset, main]);
  assert.equal(f.cache?.conflict, true);
});

test("stale local version fails before contacting or changing the cloud", async () => {
  const f = fixture(),
    sync = f.client(),
    initial = await sync.read();
  await sync.save(put({ version: initial.version, entries: [dataset, main] }));
  const count = f.calls.filter((call) => call.target === "cloud").length;
  await assert.rejects(
    sync.save(put({ version: initial.version, entries: [main, dataset] })),
    isStatus(409),
  );
  assert.equal(f.calls.filter((call) => call.target === "cloud").length, count);
});

test("viewer may read but cannot change local or shared layout", async () => {
  const f = fixture(),
    sync = f.client();
  f.viewer();
  const initial = await sync.read();
  const count = f.calls.length;
  await assert.rejects(
    sync.save(put({ version: initial.version, entries: [dataset, main] })),
    isStatus(403),
  );
  assert.equal(f.calls.length, count);
  await sync.retry();
  assert.equal(f.calls.length, count);
});

test("authorization loss disables later writes rather than masquerading as offline", async () => {
  const f = fixture(),
    sync = f.client();
  await sync.read();
  f.deny();
  await assert.rejects(sync.read(), isStatus(403));
  assert.equal(f.state?.access_denied, true);
  const count = f.calls.length;
  await assert.rejects(sync.save(put(f.local)), isStatus(403));
  assert.equal(f.calls.length, count);
});

test("an older deployment 404 is explicit and leaves local layout intact", async () => {
  const f = fixture(),
    sync = f.client();
  await sync.read();
  const previous = structuredClone(f.local);
  f.oldServer();
  await assert.rejects(sync.read(), isStatus(404));
  assert.deepEqual(f.local, previous);
});

test("an unshared local dataset preserves its layout and prevents incomplete cloud PUT", async () => {
  const f = fixture(),
    sync = f.client();
  await sync.read();
  f.addLocalDataset();
  const current = structuredClone(f.local);
  const saved = await sync.save(
    put({
      ...current,
      entries: [
        folder,
        ...current.entries.map((entry) => ({ ...entry, parent: FOLDER })),
      ],
    }),
  );
  assert.equal(saved.pending_sync, true);
  assert.equal(saved.entries.length, 4);
  assert.equal(f.remote.entries.length, 2);
  assert.equal(
    f.calls.some((call) => call.target === "cloud" && call.method === "PUT"),
    false,
  );
});

test("missing cloud data cache fails with inventory conflict without dropping remote entries", async () => {
  const f = fixture(),
    sync = f.client();
  f.unmap();
  await assert.rejects(
    sync.read(),
    (error: unknown) =>
      error instanceof ApiError && error.code === "FILE_INVENTORY_CONFLICT",
  );
  assert.deepEqual(f.local.entries, [main, dataset]);
  assert.equal(f.cache, null);
});

test("first offline edit can synchronize only against a previously untouched remote layout", async () => {
  const f = fixture(),
    sync = f.client();
  f.freshRemote();
  f.offline();
  const initial = await sync.read();
  const desired = [dataset, { ...main, name: "fresh.py" }];
  assert.equal(
    (await sync.save(put({ version: initial.version, entries: desired })))
      .pending_sync,
    true,
  );
  f.online();
  const synced = await f.client().read();
  assert.equal(synced.pending_sync, false);
  assert.deepEqual(f.remote.entries, [
    { ...dataset, id: CLOUD },
    { ...main, name: "fresh.py" },
  ]);
});

test("first offline edit cannot overwrite an already edited remote layout", async () => {
  const f = fixture(),
    sync = f.client();
  f.offline();
  const initial = await sync.read();
  await sync.save(put({ version: initial.version, entries: [dataset, main] }));
  f.online();
  const result = await f.client().read();
  assert.equal(result.sync_conflict, true);
  assert.deepEqual(f.remote.entries, [main, { ...dataset, id: CLOUD }]);
});

test("invalid metadata cannot reach local storage or cloud", async () => {
  const f = fixture(),
    sync = f.client();
  await sync.read();
  const count = f.calls.length;
  for (const document of [
    { ...f.local, code: "exec('bad')" },
    { ...f.local, version: -1 },
    { ...f.local, entries: [main, { ...dataset, id: "../outside" }] },
    { ...f.local, entries: [{ ...main, name: "../analysis.py" }, dataset] },
    { ...f.local, entries: [folder, { ...main, parent: "0".repeat(32) }] },
  ])
    await assert.rejects(sync.save(JSON.stringify(document)));
  assert.equal(f.calls.length, count);
});

test("local layout acknowledgement bypasses an in-flight executable save and preserves exact code", async () => {
  const f = hybridFixture();
  const initial = await f.hybrid.request<FileLayout>("/files/layout");
  const delayed = f.delayDraft();
  const saving = f.hybrid.request("/console/script", {
    method: "PUT",
    body: JSON.stringify({ code: "unfinished = (\n    42" }),
  });
  await delayed.started;
  const cloudCount = f.cloudCalls.length;
  const desired = [{ ...main, name: "research.py" }, dataset];
  const saved = await f.hybrid.request<FileLayout>("/files/layout", {
    method: "PUT",
    body: put({ version: initial.version, entries: desired }),
  });
  assert.deepEqual(saved.entries, desired);
  assert.equal(saved.pending_sync, true);
  assert.equal(
    f.cloudCalls.length,
    cloudCount,
    "Rename starts no network request before local acknowledgement",
  );
  assert.equal(
    f.code,
    "unfinished = (\n    42",
    "Logical rename never flushes or rewrites Python source",
  );
  const localOnly = await f.hybrid.request<FileLayout>("/files/layout", {
    headers: { "X-OpenEcon-Local-Only": "true" },
  });
  assert.equal(localOnly.version, saved.version);
  delayed.release();
  await saving;
  const synced = await f.hybrid.request<FileLayout>("/files/layout");
  assert.equal(synced.pending_sync, false);
  assert.deepEqual(synced.entries, desired);
  assert.equal(f.code, "unfinished = (\n    42");
  f.hybrid.cancelPending();
});

test("a second rename can persist during cloud upload and notifications never acknowledge its older predecessor", async () => {
  const f = hybridFixture();
  const initial = await f.hybrid.request<FileLayout>("/files/layout");
  const notices: FileLayout[] = [];
  const unsubscribe = f.hybrid.subscribeFileLayout!((value) =>
    notices.push(value),
  );
  const delayed = f.delayCloud("PUT");
  const first = await f.hybrid.request<FileLayout>("/files/layout", {
    method: "PUT",
    body: put({
      version: initial.version,
      entries: [{ ...main, name: "first.py" }, dataset],
    }),
  });
  await delayed.started;
  const secondEntries = [
    folder,
    { ...dataset, parent: FOLDER },
    { ...main, name: "second.py" },
  ];
  const second = await f.hybrid.request<FileLayout>("/files/layout", {
    method: "PUT",
    body: put({ version: first.version, entries: secondEntries }),
  });
  assert.equal(second.version, first.version + 1);
  assert.equal(second.pending_sync, true);
  delayed.release();
  await waitUntil(() => notices.some((value) => value.pending_sync === false));
  assert.ok(notices.every((value) => value.version >= second.version));
  assert.ok(
    notices.every(
      (value) =>
        JSON.stringify(value.entries) === JSON.stringify(secondEntries),
    ),
  );
  assert.ok(
    notices.some((value) => value.pending_sync === true),
    "First remote acknowledgement leaves the newer local version pending",
  );
  assert.deepEqual(notices.at(-1)?.entries, secondEntries);
  assert.equal(f.code, "original = 1");
  unsubscribe();
  f.hybrid.cancelPending();
});

test("a delayed cloud GET reconciles the latest local rename instead of restoring its stale snapshot", async () => {
  const f = fixture(),
    sync = f.client();
  const initial = await sync.read();
  const delayed = f.delayCloud("GET");
  const refresh = sync.read();
  await delayed.started;
  const desired = [{ ...main, name: "latest.py" }, dataset];
  const local = await sync.saveLocal(
    put({ version: initial.version, entries: desired }),
  );
  delayed.release();
  const result = await refresh;
  assert.equal(result.version, local.version);
  assert.deepEqual(result.entries, desired);
  assert.equal(result.pending_sync, false);
  assert.deepEqual(f.remote.entries, [
    { ...main, name: "latest.py" },
    { ...dataset, id: CLOUD },
  ]);
});

test("an acknowledgement lost during overlapping renames settles only the exact submitted cloud body", async () => {
  const f = fixture(),
    sync = f.client();
  const initial = await sync.read();
  const first = await sync.saveLocal(
    put({
      version: initial.version,
      entries: [{ ...main, name: "first.py" }, dataset],
    }),
  );
  const delayed = f.delayCloud("PUT");
  f.loseAck();
  const uploading = sync.retry();
  await delayed.started;
  const secondEntries = [dataset, { ...main, name: "second.py" }];
  const second = await sync.saveLocal(
    put({ version: first.version, entries: secondEntries }),
  );
  delayed.release();
  await uploading;
  assert.equal((await sync.readLocal()).pending_sync, true);
  await sync.retry();
  const settled = await sync.readLocal();
  assert.equal(settled.sync_conflict, false);
  assert.equal(settled.pending_sync, false);
  assert.equal(settled.version, second.version);
  assert.deepEqual(settled.entries, secondEntries);
  assert.deepEqual(f.remote.entries, [
    { ...dataset, id: CLOUD },
    { ...main, name: "second.py" },
  ]);
});

test("background teammate conflict is explicit and a known conflict blocks mutation before local CAS", async () => {
  const f = hybridFixture();
  const initial = await f.hybrid.request<FileLayout>("/files/layout");
  const notices: FileLayout[] = [];
  f.hybrid.subscribeFileLayout!((value) => notices.push(value));
  const peer = [
    { ...main, name: "peer.py" },
    { ...dataset, id: CLOUD },
  ];
  f.peer(peer);
  const desired = [dataset, { ...main, name: "mine.py" }];
  const saved = await f.hybrid.request<FileLayout>("/files/layout", {
    method: "PUT",
    body: put({ version: initial.version, entries: desired }),
  });
  await waitUntil(() => notices.some((value) => value.sync_conflict));
  assert.deepEqual(notices.at(-1)?.entries, desired);
  const before = await f.hybrid.request<FileLayout>("/files/layout", {
    headers: { "X-OpenEcon-Local-Only": "true" },
  });
  await assert.rejects(
    f.hybrid.request("/files/layout", {
      method: "PUT",
      body: put({ version: saved.version, entries: [main, dataset] }),
    }),
    isStatus(409),
  );
  const after = await f.hybrid.request<FileLayout>("/files/layout", {
    headers: { "X-OpenEcon-Local-Only": "true" },
  });
  assert.deepEqual(after, before);
  assert.deepEqual(
    f.cloudCalls.filter(
      (call) => call.path === "/files/layout" && call.method === "PUT",
    ),
    [],
  );
  f.hybrid.cancelPending();
});

test("background authorization denial is reported separately from offline pending and disables later writes", async () => {
  const f = hybridFixture();
  const initial = await f.hybrid.request<FileLayout>("/files/layout");
  const notices: FileLayout[] = [];
  f.hybrid.subscribeFileLayout!((value) => notices.push(value));
  f.deny();
  const desired = [{ ...main, name: "local.py" }, dataset];
  const saved = await f.hybrid.request<FileLayout>("/files/layout", {
    method: "PUT",
    body: put({ version: initial.version, entries: desired }),
  });
  await waitUntil(() =>
    notices.some((value) => value.sync_error?.status === 403),
  );
  assert.equal(f.state.access_denied, true);
  assert.equal(notices.at(-1)?.pending_sync, true);
  assert.equal(notices.at(-1)?.sync_conflict, false);
  assert.deepEqual(notices.at(-1)?.entries, desired);
  const count = f.localCalls.length;
  await assert.rejects(
    f.hybrid.request("/files/layout", {
      method: "PUT",
      body: put({ version: saved.version, entries: [dataset, main] }),
    }),
    isStatus(403),
  );
  assert.equal(f.localCalls.length, count);
  f.hybrid.cancelPending();
});

test("unsubscribing and closing a workspace suppress late notifications and new cloud writes", async () => {
  const f = hybridFixture();
  const initial = await f.hybrid.request<FileLayout>("/files/layout");
  const notices: FileLayout[] = [];
  const unsubscribe = f.hybrid.subscribeFileLayout!((value) =>
    notices.push(value),
  );
  const delayed = f.delayCloud("GET");
  await f.hybrid.request<FileLayout>("/files/layout", {
    method: "PUT",
    body: put({ version: initial.version, entries: [dataset, main] }),
  });
  await delayed.started;
  unsubscribe();
  f.hybrid.cancelPending();
  delayed.release();
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.deepEqual(notices, []);
  assert.equal(
    f.cloudCalls.some(
      (call) => call.path === "/files/layout" && call.method === "PUT",
    ),
    false,
  );
});

test("transient offline background sync preserves durable pending names without a hard error", async () => {
  const f = hybridFixture();
  const initial = await f.hybrid.request<FileLayout>("/files/layout");
  const notices: FileLayout[] = [];
  f.hybrid.subscribeFileLayout!((value) => notices.push(value));
  f.offline();
  const desired = [dataset, { ...main, name: "offline.py" }];
  await f.hybrid.request("/files/layout", {
    method: "PUT",
    body: put({ version: initial.version, entries: desired }),
  });
  await waitUntil(() => notices.length > 0);
  assert.equal(notices.at(-1)?.pending_sync, true);
  assert.equal(notices.at(-1)?.sync_error, undefined);
  assert.equal(notices.at(-1)?.sync_conflict, false);
  assert.deepEqual(
    (await f.client().read()).entries,
    desired,
    "A new adapter reads the same durable offline layout",
  );
  f.hybrid.cancelPending();
});

test("unsupported and inventory failures are explicit background errors while local names remain pending", async () => {
  for (const scenario of ["unsupported", "inventory"] as const) {
    const f = hybridFixture();
    const initial = await f.hybrid.request<FileLayout>("/files/layout");
    const notices: FileLayout[] = [];
    f.hybrid.subscribeFileLayout!((value) => notices.push(value));
    if (scenario === "unsupported") f.oldServer();
    else f.inventoryConflict();
    const desired = [dataset, { ...main, name: "retained.py" }];
    await f.hybrid.request("/files/layout", {
      method: "PUT",
      body: put({ version: initial.version, entries: desired }),
    });
    await waitUntil(() => notices.some((value) => value.sync_error));
    const latest = notices.at(-1)!;
    assert.equal(
      latest.sync_error?.status,
      scenario === "unsupported" ? 404 : 409,
    );
    assert.equal(
      latest.sync_error?.code,
      scenario === "unsupported" ? "NOT_FOUND" : "FILE_INVENTORY_CONFLICT",
    );
    assert.equal(latest.pending_sync, true);
    assert.equal(latest.sync_conflict, false);
    assert.deepEqual(latest.entries, desired);
    f.hybrid.cancelPending();
  }
});

test("pending new scripts synchronize before their logical layout without renaming canonical source", async () => {
  const f = hybridFixture();
  // Start from an already persisted layout base, as after reopening a project
  // containing a file created during the previous offline session.
  const initial = await f.client().read();
  const script = f.pendingScript("e".repeat(32));
  const desired = [
    ...initial.entries,
    {
      kind: "script" as const,
      id: script.id,
      name: "renamed.py",
      parent: null,
    },
  ];
  const notices: FileLayout[] = [];
  f.hybrid.subscribeFileLayout!((value) => notices.push(value));
  const saved = await f.hybrid.request<FileLayout>("/files/layout", {
    method: "PUT",
    body: put({ version: initial.version, entries: desired }),
  });
  assert.equal(saved.pending_sync, true);
  await waitUntil(() => notices.some((value) => value.pending_sync === false));
  const createdAt = f.cloudCalls.findIndex(
    (call) => call.path === "/console/scripts" && call.method === "POST",
  );
  const layoutAt = f.cloudCalls.findIndex(
    (call) => call.path === "/files/layout" && call.method === "PUT",
  );
  assert.ok(createdAt >= 0 && layoutAt > createdAt);
  assert.equal(f.cloudCalls[createdAt].body.name, "pending.py");
  assert.equal(f.cloudCalls[createdAt].body.code, "unsaved = (");
  assert.equal(f.cloudCalls[layoutAt].body.entries.at(-1).name, "renamed.py");
  assert.deepEqual(notices.at(-1)?.entries, desired);
  assert.equal(
    f.cloudCalls.some((call) => call.path === "/console/script"),
    false,
  );
  assert.equal(f.code, "original = 1");
  f.hybrid.cancelPending();
});

test("coalesced rename uploads yield to executable saves already waiting in the remote queue", async () => {
  const f = hybridFixture();
  const initial = await f.hybrid.request<FileLayout>("/files/layout");
  const notices: FileLayout[] = [];
  f.hybrid.subscribeFileLayout!((value) => notices.push(value));
  const delayed = f.delayCloud("PUT");
  const first = await f.hybrid.request<FileLayout>("/files/layout", {
    method: "PUT",
    body: put({
      version: initial.version,
      entries: [{ ...main, name: "first.py" }, dataset],
    }),
  });
  await delayed.started;
  const desired = [dataset, { ...main, name: "last.py" }];
  await f.hybrid.request<FileLayout>("/files/layout", {
    method: "PUT",
    body: put({ version: first.version, entries: desired }),
  });
  const source = f.hybrid.request("/console/script", {
    method: "PUT",
    body: JSON.stringify({ code: "current = 42" }),
  });
  delayed.release();
  await source;
  await waitUntil(() => notices.some((value) => value.pending_sync === false));
  const uploads = f.cloudCalls.flatMap((call, index) =>
    call.path === "/files/layout" && call.method === "PUT" ? [index] : [],
  );
  const sourceAt = f.cloudCalls.findIndex(
    (call) => call.path === "/console/script",
  );
  assert.equal(uploads.length, 2);
  assert.ok(
    uploads[0] < sourceAt && sourceAt < uploads[1],
    "Next metadata pass goes to the tail after waiting source work",
  );
  assert.equal(f.code, "current = 42");
  assert.deepEqual(notices.at(-1)?.entries, desired);
  f.hybrid.cancelPending();
});

test("authorization denied during an online-event retry reaches the layout subscriber", async () => {
  const previousWindow = globalThis.window;
  globalThis.window = new EventTarget() as unknown as Window &
    typeof globalThis;
  const f = hybridFixture();
  try {
    const initial = await f.hybrid.request<FileLayout>("/files/layout");
    const notices: FileLayout[] = [];
    f.hybrid.subscribeFileLayout!((value) => notices.push(value));
    f.offline();
    await f.hybrid.request("/files/layout", {
      method: "PUT",
      body: put({ version: initial.version, entries: [dataset, main] }),
    });
    await waitUntil(() => notices.length > 0);
    f.online();
    f.deny();
    window.dispatchEvent(new Event("online"));
    await waitUntil(() =>
      notices.some((value) => value.sync_error?.status === 403),
    );
    assert.equal(f.state.access_denied, true);
    assert.deepEqual(notices.at(-1)?.entries, [dataset, main]);
  } finally {
    f.hybrid.cancelPending();
    globalThis.window = previousWindow;
  }
});

test("a legacy Python-only alias rejection leaves the durable local name pending and retries after cloud upgrade", async () => {
  for (const name of ["notes.md", "table.tex"]) {
    const f = fixture(), sync = f.client();
    const named: FileLayoutEntry = { kind: "script", id: "b".repeat(32), name: "code.py", parent: null };
    f.addLocalScript(named);
    f.shareScript(named);
    await sync.read();
    f.legacyDocuments();
    const saved = await sync.saveLocal(put({ ...f.local, entries: f.local.entries.map(entry => entry.id === named.id ? { ...entry, name } : entry) }));
    assert.equal(saved.pending_sync, true);
    await assert.rejects(sync.retry(), failure => failure instanceof ApiError && failure.status === 422 && failure.code === "INVALID_FILE_LAYOUT");
    assert.equal(f.local.entries.find(entry => entry.id === named.id)?.name, name);
    assert.equal(f.remote.entries.find(entry => entry.id === named.id)?.name, "code.py");
    assert.equal((await sync.readLocal()).pending_sync, true);
    assert.equal(f.cache?.conflict, false);
    f.documentUpgrade();
    await sync.retry();
    assert.equal(f.remote.entries.find(entry => entry.id === named.id)?.name, name);
    assert.equal((await sync.readLocal()).pending_sync, false);
  }
});

const privateData: FileLayoutEntry = { kind: "dataset", id: "33333333-3333-4333-8333-333333333333", name: "large.csv", parent: null };
test("explicit local-only data survives remote adoption and is omitted from shared layout writes", async () => {
  const f = fixture(), sync = f.client(); f.addPrivateDataset(privateData);
  const opened = await sync.read();
  assert.deepEqual(opened.entries, [main, dataset, privateData]);
  assert.equal(opened.pending_sync, false);
  const renamed = { ...privateData, name: "local-large.csv" };
  const sharedMain = { ...main, name: "shared.py" };
  await sync.save(put({ ...f.local, entries: [renamed, sharedMain, dataset] }));
  assert.deepEqual(f.local.entries, [renamed, sharedMain, dataset]);
  assert.deepEqual(f.remote.entries, [sharedMain, { ...dataset, id: CLOUD }]);
  assert.ok(f.calls.filter(call => call.target === "cloud" && call.method === "PUT").every(call =>
    !call.body.entries.some((entry: FileLayoutEntry) => entry.id === privateData.id)));
  const reopened = await f.client().read();
  assert.deepEqual(reopened.entries, [renamed, sharedMain, dataset]);
  assert.equal(reopened.pending_sync, false);
});

test("a local-only rename does not publish an incomplete data inventory or block script sharing", async () => {
  const f = fixture(), sync = f.client(); f.addPrivateDataset(privateData); await sync.read();
  const renamed = { ...privateData, name: "renamed.csv" };
  const saved = await sync.save(put({ ...f.local, entries: [main, dataset, renamed] }));
  assert.equal(saved.pending_sync, false);
  assert.equal(f.calls.some(call => call.target === "cloud" && call.method === "PUT"), false);
  const namedMain = { ...main, name: "model.py" };
  await sync.save(put({ ...f.local, entries: [namedMain, dataset, renamed] }));
  assert.equal(f.remote.entries[0].name, "model.py");
  assert.deepEqual(f.local.entries.at(-1), renamed);
});

test("peer ordering preserves local-only sibling order and exact private names", async () => {
  const f = fixture(), sync = f.client();
  const second = { ...privateData, id: "44444444-4444-4444-8444-444444444444", name: "second.csv" };
  f.addPrivateDataset(privateData); f.addPrivateDataset(second); await sync.read();
  await sync.save(put({ ...f.local, entries: [main, privateData, dataset, second] }));
  f.peer([{ ...dataset, id: CLOUD }, { ...main, name: "peer.py" }]);
  const adopted = await sync.read();
  assert.equal(adopted.sync_conflict, false);
  assert.deepEqual(adopted.entries.filter(entry => entry.kind === "dataset" && entry.id !== DATA), [privateData, second]);
  assert.equal(adopted.entries.find(entry => entry.id === "analysis")?.name, "peer.py");
});

test("a peer cannot overwrite a colliding local-only name", async () => {
  const f = fixture(), sync = f.client(); f.addPrivateDataset(privateData); await sync.read();
  const before = structuredClone(f.local);
  f.peer([main, { ...dataset, id: CLOUD, name: privateData.name }]);
  await assert.rejects(sync.read(), error => error instanceof ApiError && error.code === "FILE_INVENTORY_CONFLICT");
  assert.deepEqual(f.local, before);
});

test("a remotely removed folder cannot erase its local-only dataset", async () => {
  const f = fixture(), sync = f.client(); await sync.read();
  await sync.save(put({ ...f.local, entries: [main, dataset, folder] }));
  const nested = { ...privateData, parent: folder.id }; f.addPrivateDataset(nested);
  await sync.read(); const before = structuredClone(f.local);
  f.peer([main, { ...dataset, id: CLOUD }]);
  await assert.rejects(sync.read(), error => error instanceof ApiError && error.code === "FILE_INVENTORY_CONFLICT");
  assert.deepEqual(f.local, before);
});
