import test from "node:test";
import assert from "node:assert/strict";
import { ApiError, type WorkspaceClient } from "../src/api.ts";
import { createHybridWorkspace } from "../src/desktop-sync.ts";
import type { PackageManifest } from "../src/package-types.ts";

function fixture(role: "owner" | "viewer" = "owner", modern = false) {
  const base = { python: "3.13", core: { torch: "2.14.0" },
    requirements: [{ name: "humanize", version: "4.14.0" }], locked: [{ name: "humanize", version: "4.14.0" }] };
  const manifest: PackageManifest = modern
    ? { ...base, schema: 2, installer: "uv", specifications: ['humanize[example]>=4,<5; platform_system == "Darwin"'] }
    : { ...base, schema: 1 };
  let version = 0, online = true, revoked = false;
  let remote: PackageManifest | null = null;
  const installs: unknown[] = [], cloudWrites: unknown[] = [];
  const local: WorkspaceClient = {
    projectId: null, teams: false, connect: async () => ({ token: "local", version: "test", environment: "local" }),
    cancelPending() {},
    async request<T>(path, options = {}) {
      if (path === "/desktop-sync-state") return { role, base_code: "", cloud_version: 0 } as T;
      if (path === "/environment") return { manifest, job: null } as T;
      if (path === "/environment/restore") { installs.push(JSON.parse(String(options.body))); return { accepted: true } as T; }
      throw new Error(`Unexpected local route ${path}`);
    },
  };
  const client = createHybridWorkspace("a".repeat(32), role, {
    local, actorUid: "own-test", cacheFile: async () => { throw new Error("No files"); },
    async cloud<T>(path, options = {}) {
      assert.equal(path, "/environment");
      if (revoked) throw new ApiError("Access revoked", 403);
      if (!online) throw new TypeError("offline");
      if (options.method === "PUT") {
        const body = JSON.parse(String(options.body));
        if (body.version !== version) throw new ApiError("Conflict", 409, "VERSION_CONFLICT");
        cloudWrites.push(body); remote = body.manifest; version++;
      }
      return structuredClone({ manifest: remote, version }) as T;
    },
  });
  return { client, installs, cloudWrites, offline() { online = false; }, revoke() { revoked = true; },
    concurrentWrite() { remote = manifest; version++; } };
}

test("shared metadata reading never installs packages; explicit restore runs only locally", async () => {
  const f = fixture(); f.concurrentWrite();
  const shared = await f.client.request<{ localMatches: boolean }>("/environment/shared");
  assert.equal(shared.localMatches, true);
  assert.equal(f.installs.length, 0);
  await f.client.request("/environment/restore-shared", { method: "POST" });
  assert.equal(f.installs.length, 1); assert.equal(f.cloudWrites.length, 0);
});

test("team sharing uses optimistic versions and preserves a concurrent update", async () => {
  const f = fixture(); await f.client.request("/environment/shared");
  f.concurrentWrite();
  await assert.rejects(f.client.request("/environment/share", { method: "POST" }),
    (error) => error instanceof ApiError && error.code === "VERSION_CONFLICT");
  assert.equal(f.cloudWrites.length, 0); assert.equal(f.installs.length, 0);
});

test("viewer cannot share, offline restore cannot install, revocation is not treated as offline", async () => {
  const viewer = fixture("viewer");
  await assert.rejects(viewer.client.request("/environment/share", { method: "POST" }),
    (error) => error instanceof ApiError && error.status === 403);
  const f = fixture(); await f.client.request("/environment/shared"); f.offline();
  assert.equal((await f.client.request<{ offline: boolean }>("/environment/shared")).offline, true);
  await assert.rejects(f.client.request("/environment/restore-shared", { method: "POST" }));
  assert.equal(f.installs.length, 0);
  const denied = fixture(); denied.revoke();
  await assert.rejects(denied.client.request("/environment/shared"), (error) => error instanceof ApiError && error.status === 403);
});

test("consecutive team shares serialize and advance the acknowledged version", async () => {
  const f = fixture();
  await Promise.all([f.client.request("/environment/share", { method: "POST" }), f.client.request("/environment/share", { method: "POST" })]);
  assert.deepEqual(f.cloudWrites.map((body: any) => body.version), [0, 1]);
});

test("uv declarations survive team sharing and explicit local restore without losing extras or markers", async () => {
  const f = fixture("owner", true);
  await f.client.request("/environment/share", { method: "POST" });
  assert.equal((f.cloudWrites[0] as any).manifest.schema, 2);
  assert.equal((f.cloudWrites[0] as any).manifest.installer, "uv");
  assert.deepEqual((f.cloudWrites[0] as any).manifest.specifications,
    ['humanize[example]>=4,<5; platform_system == "Darwin"']);
  assert.equal(f.installs.length, 0);
  await f.client.request("/environment/restore-shared", { method: "POST" });
  assert.deepEqual((f.installs[0] as any).manifest, (f.cloudWrites[0] as any).manifest);
});
