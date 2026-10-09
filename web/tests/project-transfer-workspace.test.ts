import assert from "node:assert/strict";
import test from "node:test";
import { ApiError, type WorkspaceClient } from "../src/api.ts";
import {
  createHybridWorkspace,
  createNativeTransferInvoke,
} from "../src/desktop-sync.ts";
import type { DatasetProfile } from "../src/types.ts";
import type { CachedCloudFile } from "../src/desktop.ts";
const pid = "1".repeat(32),
  request = "a".repeat(32),
  cloudId = "b".repeat(32),
  sha = "c".repeat(64);
const view = {
  request_id: request,
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
const localData = {
  id: "33333333-3333-4333-8333-333333333333",
  name: view.name,
  data_hash: "d".repeat(64),
  local_only: true,
  size_bytes: 400,
} as DatasetProfile;
const completed = {
  ...localData,
  id: "44444444-4444-4444-8444-444444444444",
  cloud_id: cloudId,
  sha256: sha,
  local_only: false,
};
function fixture(
  download?: Promise<CachedCloudFile>,
  bootstrapFile?: "chunked" | "legacy",
  cachePinOverride?: Partial<CachedCloudFile>,
) {
  const calls: { path: string; body?: Record<string, unknown> }[] = [],
    bridges: string[] = [];
  let state: unknown = null,
    pending: unknown = view,
    deny = false,
    viewer = false;
  const data: DatasetProfile[] = [];
  const local: WorkspaceClient = {
    projectId: null,
    teams: false,
    cancelPending() {},
    async connect() {
      return { token: "local", version: "0.3", environment: "local" };
    },
    async request<T>(path, options = {}) {
      const body = options.body ? JSON.parse(String(options.body)) : undefined;
      calls.push({ path, body });
      let result: unknown;
      if (path === "/desktop-sync-state") {
        if (options.method === "PUT") state = body;
        result = state;
      } else if (path === "/console/script")
        result = { code: "x=1", name: "analysis.py" };
      else if (path === "/console")
        result = {
          history: [],
          variables: [],
          status: { running: false, session_generation: 1 },
        };
      else if (path === "/datasets") result = { datasets: data };
      else if (path === `/datasets/${localData.id}`) result = localData;
      else if (path === "/datasets/import-cached") {
        data.push(completed);
        result = completed;
      } else if (path === "/desktop-sharing")
        result = { records: [], pending_count: 0, failed_count: 0 };
      else if (path === "/desktop-outbox") result = { items: [] };
      else if (path === "/desktop-cached-files") result = { files: [] };
      else throw new Error(`Unexpected local ${path}`);
      return structuredClone(result) as T;
    },
  };
  const client = createHybridWorkspace(pid, "editor", {
    actorUid: "qa",
    local,
    async cloud<T>(path) {
      if (deny) throw new ApiError("denied", 403);
      if (path === "/bootstrap")
        return {
          session: { read_only: viewer },
          draft: { code: "x=1", version: 0, name: "analysis.py" },
          datasets: bootstrapFile
            ? [
                {
                  ...file,
                  ...(bootstrapFile === "legacy" ? { transfer: "legacy" } : {}),
                },
              ]
            : [],
          status: { running: false },
        } as T;
      throw new Error(`Unexpected cloud ${path}`);
    },
    async cacheFile(value) {
      bridges.push(`cache:${value.id}`);
      assert.deepEqual(
        value,
        bootstrapFile === "legacy" ? { ...file, transfer: "legacy" } : file,
      );
      if (download) return download;
      return {
        cloud_id: cloudId,
        sha256: sha,
        name: view.name,
        python_path: view.name,
        data_hash: sha,
        size_bytes: view.size_bytes,
        reused: true,
        ...cachePinOverride,
      } as CachedCloudFile;
    },
    async cancelFileDownload(id) {
      bridges.push(`cancel_download:${id}`);
    },
    async cacheFiles() {
      return [];
    },
    async chooseAndUpload() {
      data.push(localData);
      return { ...localData, sharing_pending_transfer: pending };
    },
    async transferInvoke(action, id) {
      bridges.push(`${action}:${id || ""}`);
      return action === "list"
        ? { status: 200, body: { transfers: [view] } }
        : action === "cancel"
          ? { status: 409, body: { detail: { code: "TRANSFER_COMPLETE" } } }
          : { status: 200, body: { state: "ready", file } };
    },
  });
  return {
    client,
    calls,
    bridges,
    data,
    badPending() {
      pending = { ...view, token: "secret" };
    },
    deny() {
      deny = true;
    },
    viewer() {
      viewer = true;
    },
  };
}
test("interrupted upload returns existing local-owned dataset plus public pending view without cloud download", async () => {
  const h = fixture();
  await h.client.request("/bootstrap");
  const result = await h.client.request<
    DatasetProfile & { sharing_pending_transfer: unknown }
  >("/datasets/upload", { method: "POST" });
  assert.equal(result.id, localData.id);
  assert.deepEqual(result.sharing_pending_transfer, view);
  assert.deepEqual(h.bridges, []);
  assert.ok(!h.calls.some((call) => call.path === "/datasets/import-cached"));
  assert.ok(!JSON.stringify(h.calls).includes("sharing_pending_transfer"));
});
test("malformed pending metadata cannot smuggle a token or local path into the workspace response", async () => {
  const h = fixture();
  await h.client.request("/bootstrap");
  h.badPending();
  await assert.rejects(
    h.client.request("/datasets/upload", { method: "POST" }),
    /verified/,
  );
  assert.equal(h.data[0].id, localData.id);
  assert.deepEqual(h.bridges, []);
});
test("explicit resume caches exact pins and returns fresh cloud-linked local profile while preserving original and draft", async () => {
  const h = fixture();
  await h.client.request("/bootstrap");
  await h.client.request("/datasets/upload", { method: "POST" });
  const before = h.calls.filter(
    (call) => call.path === "/console/script" && call.body,
  ).length;
  await h.client.transferActions!.refresh();
  const ready = await h.client.transferActions!.resume(request);
  assert.deepEqual(ready, completed);
  assert.deepEqual(h.bridges, [
    "list:",
    `resume:${request}`,
    `cache:${cloudId}`,
  ]);
  assert.deepEqual(
    h.calls.find((call) => call.path === "/datasets/import-cached")!.body,
    { name: view.name, expected_sha256: sha, cloud_id: cloudId },
  );
  assert.equal(h.data.length, 2);
  assert.equal(h.data[0].id, localData.id);
  assert.equal(
    h.calls.filter((call) => call.path === "/console/script" && call.body)
      .length,
    before,
  );
});
test("project close, viewer downgrade and revocation block native transfer mutations", async () => {
  for (const mode of ["close", "viewer", "deny"]) {
    const h = fixture();
    await h.client.request("/bootstrap");
    await h.client.transferActions!.refresh();
    if (mode === "close") h.client.cancelPending();
    else if (mode === "viewer") {
      h.viewer();
      await h.client.request("/bootstrap");
    } else {
      h.deny();
      await assert.rejects(h.client.request("/bootstrap"));
    }
    await assert.rejects(h.client.transferActions!.resume(request));
    assert.deepEqual(h.bridges, ["list:"]);
  }
});
test("native transfer authentication is token-closure bound and refreshes only one 401", async () => {
  for (const thrown of [false, true]) {
    const tokens: boolean[] = [],
      calls: Record<string, unknown>[] = [];
    const invoke = createNativeTransferInvoke(
      pid,
      async (force) => {
        tokens.push(!!force);
        return force ? "renewed" : "initial";
      },
      async <T>(command, args) => {
        assert.equal(command, "project_transfer_action");
        calls.push(args!);
        if (calls.length === 1) {
          if (thrown) throw new ApiError("expired", 401);
          return { status: 401, body: {} } as T;
        }
        return { status: 200, body: { transfers: [] } } as T;
      },
    );
    assert.equal((await invoke("list")).status, 200);
    assert.deepEqual(tokens, [false, true]);
    assert.deepEqual(
      calls.map((x) => x.token),
      ["initial", "renewed"],
    );
    assert.deepEqual(
      calls.map((x) => [x.projectId, x.requestId, x.action]),
      [
        [pid, null, "list"],
        [pid, null, "list"],
      ],
    );
  }
});
test("native permissions, integrity and unavailable-service errors do not refresh credentials or silently resume", async () => {
  for (const status of [403, 404, 422]) {
    let calls = 0;
    const invoke = createNativeTransferInvoke(
      pid,
      async () => {
        calls++;
        return "fixture";
      },
      async <T>() => {
        throw new ApiError("failure", status);
      },
    );
    await assert.rejects(invoke("resume", request));
    assert.equal(calls, 1);
  }
});

test("exact native completed sentinel remains a typed public refusal after a refreshed token", async () => {
  let calls = 0;
  const invoke = createNativeTransferInvoke(
    pid,
    async () => "fixture",
    async <T>() => {
      if (++calls === 1) return { status: 401, body: {} } as T;
      throw "OPENECON_TRANSFER_COMPLETE";
    },
  );
  await assert.rejects(
    invoke("cancel", request),
    (error) =>
      error instanceof ApiError &&
      error.status === 409 &&
      error.code === "TRANSFER_COMPLETE",
  );
  assert.equal(calls, 2);
});

test("active cloud download is visible to late subscribers and cancellation preserves original/local draft", async () => {
  let reject!: (reason: unknown) => void;
  const pending = new Promise<CachedCloudFile>((_resolve, fail) => {
    reject = fail;
  });
  const h = fixture(pending);
  await h.client.request("/bootstrap");
  await h.client.request("/datasets/upload", { method: "POST" });
  await h.client.transferActions!.refresh();
  const views: ({ id: string; name: string } | null)[] = [];
  const unsubscribe = h.client.subscribeFileDownload!((value) =>
    views.push(value),
  );
  const resume = h.client.transferActions!.resume(request);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(views, [null, { id: cloudId, name: view.name }]);
  let late: unknown;
  const removeLate = h.client.subscribeFileDownload!((value) => {
    late = value;
    if (value) value.name = "observer mutation";
  });
  assert.ok(late);
  await h.client.cancelFileDownload!(cloudId);
  assert.equal(h.bridges.at(-1), `cancel_download:${cloudId}`);
  reject("OPENECON_DOWNLOAD_CANCELLED");
  await assert.rejects(
    resume,
    (error) => error instanceof DOMException && error.name === "AbortError",
  );
  assert.equal(views.at(-1), null);
  assert.equal(h.data.length, 1);
  assert.equal(h.data[0].id, localData.id);
  assert.ok(!h.calls.some((call) => call.path === "/datasets/import-cached"));
  assert.equal(h.client.transferActions!.getSnapshot().transfers.length, 1);
  await assert.rejects(
    h.client.cancelFileDownload!(cloudId),
    /no longer downloading/,
  );
  unsubscribe();
  removeLate();
});
test("closed project ignores a late cached download before importing into the local worker", async () => {
  let resolve!: (value: CachedCloudFile) => void;
  const pending = new Promise<CachedCloudFile>((done) => {
    resolve = done;
  });
  const h = fixture(pending);
  await h.client.request("/bootstrap");
  await h.client.transferActions!.refresh();
  const resume = h.client.transferActions!.resume(request);
  await new Promise((done) => setImmediate(done));
  h.client.cancelPending();
  resolve({
    cloud_id: cloudId,
    sha256: sha,
    name: view.name,
    python_path: view.name,
    data_hash: sha,
    size_bytes: view.size_bytes,
    reused: true,
  });
  await assert.rejects(resume, /Workspace closed/);
  assert.ok(!h.calls.some((call) => call.path === "/datasets/import-cached"));
});
test("local download cancellation uses exactly the same bound project token and canonical file ID", async () => {
  let actual: unknown;
  const invoke = createNativeTransferInvoke(
    pid,
    async () => "private-fixture",
    async <T>(command, args) => {
      actual = { command, args };
      return {
        status: 200,
        body: { cancel_requested: true, file_id: cloudId },
      } as T;
    },
  );
  assert.deepEqual((await invoke("cancel_download", cloudId)).body, {
    cancel_requested: true,
    file_id: cloudId,
  });
  assert.deepEqual(actual, {
    command: "project_transfer_action",
    args: {
      projectId: pid,
      requestId: cloudId,
      token: "private-fixture",
      action: "cancel_download",
    },
  });
});

test("legacy file downloads have no chunk cancellation UI or action", async () => {
  const h = fixture(undefined, "legacy");
  const views: unknown[] = [];
  h.client.subscribeFileDownload!((value) => views.push(value));
  await h.client.request("/bootstrap");
  assert.deepEqual(views, [null]);
  await assert.rejects(
    h.client.cancelFileDownload!(cloudId),
    /no longer downloading/,
  );
});
test("a current viewer may cancel an active chunked download during project bootstrap", async () => {
  let reject!: (error: unknown) => void;
  const pending = new Promise<CachedCloudFile>((_resolve, fail) => {
    reject = fail;
  });
  const h = fixture(pending, "chunked");
  h.viewer();
  const views: unknown[] = [];
  h.client.subscribeFileDownload!((value) => views.push(value));
  const bootstrap = h.client.request("/bootstrap");
  await new Promise((done) => setImmediate(done));
  assert.deepEqual(views, [null, { id: cloudId, name: view.name }]);
  await h.client.cancelFileDownload!(cloudId);
  assert.equal(h.bridges.at(-1), `cancel_download:${cloudId}`);
  reject("OPENECON_DOWNLOAD_CANCELLED");
  await assert.rejects(
    bootstrap,
    (error) => error instanceof DOMException && error.name === "AbortError",
  );
  assert.ok(!h.calls.some((call) => call.path === "/datasets/import-cached"));
});

test("upload cancellation while ready cache is downloading prevents a later worker import", async () => {
  let resolve!: (value: CachedCloudFile) => void;
  const pending = new Promise<CachedCloudFile>((done) => {
    resolve = done;
  });
  const h = fixture(pending);
  await h.client.request("/bootstrap");
  await h.client.request("/datasets/upload", { method: "POST" });
  await h.client.transferActions!.refresh();
  const resume = h.client.transferActions!.resume(request);
  await new Promise((done) => setImmediate(done));
  await assert.rejects(
    h.client.transferActions!.cancel(request),
    (error) => error instanceof ApiError && error.code === "TRANSFER_COMPLETE",
  );
  resolve({
    cloud_id: cloudId,
    sha256: sha,
    name: view.name,
    python_path: view.name,
    data_hash: sha,
    size_bytes: view.size_bytes,
    reused: true,
  });
  await assert.rejects(
    resume,
    (error) => error instanceof DOMException && error.name === "AbortError",
  );
  assert.ok(!h.calls.some((call) => call.path === "/datasets/import-cached"));
  assert.equal(h.data.length, 1);
  assert.equal(
    h.client.transferActions!.getSnapshot().transfers[0].state,
    "ready",
  );
});

test("mismatched chunk cache filename or physical size is refused before any worker import", async () => {
  for (const mismatch of [
    { name: "different.csv" },
    { size_bytes: view.size_bytes + 1 },
  ]) {
    const h = fixture(undefined, undefined, mismatch);
    await h.client.request("/bootstrap");
    await h.client.request("/datasets/upload", { method: "POST" });
    await h.client.transferActions!.refresh();
    await assert.rejects(
      h.client.transferActions!.resume(request),
      (error) => error instanceof ApiError && error.code === "DATA_INTEGRITY",
    );
    assert.ok(!h.calls.some((call) => call.path === "/datasets/import-cached"));
    assert.equal(h.data.length, 1);
  }
});
