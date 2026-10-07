import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { ApiError, type WorkspaceClient } from "../src/api.ts";
import { cloudArchiveRecord } from "../src/desktop-sync.ts";
import { createResultSharing, type ResultOutboxItem, type ResultSharingRecord,
  type ResultSharingSnapshot } from "../src/result-sharing.ts";
import type { ExecutionRecord } from "../src/types.ts";

const actor = "account-a";
const resultId = (name: string) => {
  const hash = createHash("sha256").update(name).digest("hex");
  return `${hash.slice(0, 8)}-${hash.slice(8, 12)}-4${hash.slice(13, 16)}-a${hash.slice(17, 20)}-${hash.slice(20, 32)}`;
};
const cloudId = (localId: string, actorUid = actor) => createHash("sha256").update(`${actorUid}:${localId}`).digest("hex").slice(0, 32);
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
async function eventually(predicate: () => boolean) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (predicate()) return;
    await tick();
  }
  assert.ok(predicate(), "background delivery did not settle");
}
function execution(id: string, actor_uid = actor): ExecutionRecord {
  return { id: resultId(id), actor_uid, code: "print(42)", status: "ok", stdout: "42\n", outputs: [],
    variables: [], events: [{ type: "stdout", text: "42\n" }], duration_ms: 1,
    session_generation: 1 };
}
function fixture() {
  const metadata = new Map<string, ResultSharingRecord>();
  const names = new Map<string, string>();
  const history = new Map<string, ResultOutboxItem>();
  const outbox = new Map<string, ResultOutboxItem>();
  const accepted = new Map<string, string>();
  const cloudCalls: { id: string; body: string }[] = [];
  const localCalls: { path: string; body?: Record<string, unknown> }[] = [];
  let denied = false, closed = false, role: "editor" | "viewer" = "editor";
  let cloudFailure: unknown;
  let acknowledgementFailure: unknown;
  let metadataFailure: unknown;
  let statusWriteFailure: unknown;
  let denialPersistenceFailure: unknown;
  let queueFull = false, oversized = false;
  let cloudWait: Promise<void> | null = null;
  let metadataWait: Promise<void> | null = null;
  let acknowledgementOverride: unknown;
  let overrideAcknowledgement = false;
  const notifications: ResultSharingSnapshot[] = [];
  const local: WorkspaceClient = {
    projectId: null, teams: false, connect: async () => ({ token: "local", version: "0.3", environment: "local", persistent: true }),
    cancelPending() {},
    async request<T>(path, options = {}) {
      const body = options.body ? JSON.parse(String(options.body)) : undefined;
      localCalls.push({ path, body });
      if (path === "/desktop-sharing") {
        if (metadataFailure) throw metadataFailure;
        const captured = structuredClone({ records: [...metadata.values()], pending_count: 999, failed_count: 999 });
        if (metadataWait) { const wait = metadataWait; metadataWait = null; await wait; }
        return captured as T;
      }
      if (path === "/desktop-outbox") return { items: [...outbox.values()].map((item) => structuredClone(item)) } as T;
      const update = /^\/desktop-sharing\/([^/]+)$/.exec(path);
      if (update) {
        if (statusWriteFailure) throw statusWriteFailure;
        const record = metadata.get(names.get(decodeURIComponent(update[1])) ?? decodeURIComponent(update[1]))!;
        assert.equal(body.actor_uid, record.actor_uid);
        if (record.state === "shared") throw new ApiError("Already shared", 409, "ALREADY_SHARED");
        metadata.set(names.get(record.id) ?? record.id, { ...record, state: body.state, error: body.error, retryable: true });
        return {} as T;
      }
      const ack = /^\/desktop-outbox\/([^?]+)\?actor_uid=(.+)$/.exec(path);
      if (ack) {
        const id = names.get(decodeURIComponent(ack[1])) ?? decodeURIComponent(ack[1]);
        assert.equal(decodeURIComponent(ack[2]), metadata.get(id)!.actor_uid);
        if (acknowledgementFailure) {
          const error = acknowledgementFailure; acknowledgementFailure = undefined; throw error;
        }
        metadata.set(id, { ...metadata.get(id)!, state: "shared", retryable: false, error: undefined });
        outbox.delete(id);
        return { deleted: true } as T;
      }
      const retry = /^\/desktop-sharing\/([^/]+)\/retry$/.exec(path);
      if (retry) {
        const id = names.get(decodeURIComponent(retry[1])) ?? decodeURIComponent(retry[1]), record = metadata.get(id)!;
        if (!record || record.actor_uid !== body.actor_uid || record.state === "local_only")
          throw new ApiError("Unavailable result", 403, "ROLE_REQUIRED");
        if (queueFull || oversized) {
          const error = { code: oversized ? "OUTBOX_LIMIT" : "OUTBOX_FULL",
            message: oversized ? "Result is too large" : "Queue is full", status: oversized ? 413 : 409 };
          metadata.set(id, { ...record, state: "failed", error, retryable: true });
          throw new ApiError(error.message, error.status, error.code);
        }
        if (record.state !== "shared") {
          outbox.set(id, structuredClone(history.get(id)!));
          metadata.set(id, { ...record, state: "pending", error: undefined, retryable: true });
        }
        return {} as T;
      }
      throw new Error(`Unexpected local request ${path}`);
    },
  };
  const dependencies = {
    actorUid: actor, local, closed: () => closed,
    accessible(editing = false) {
      if (closed) throw new DOMException("Closed", "AbortError");
      if (denied || editing && role === "viewer") throw new ApiError("Access revoked", 403, "ROLE_REQUIRED");
    },
    async deny() { denied = true; if (denialPersistenceFailure) throw denialPersistenceFailure; }, archive: cloudArchiveRecord,
    async cloud<T>(_path: string, options: RequestInit = {}) {
      const value = JSON.parse(String(options.body)) as ResultOutboxItem;
      const encoded = String(options.body);
      cloudCalls.push({ id: names.get(value.record.id) ?? value.record.id, body: encoded });
      if (cloudWait) { const wait = cloudWait; cloudWait = null; await wait; }
      if (cloudFailure) throw cloudFailure;
      const remoteId = cloudId(value.record.id);
      if (accepted.has(remoteId)) assert.equal(accepted.get(remoteId), encoded);
      accepted.set(remoteId, encoded);
      if (overrideAcknowledgement) return acknowledgementOverride as T;
      return { shared: true, id: remoteId } as T;
    },
  };
  let publisher = createResultSharing(dependencies);
  publisher.subscribe((value) => { notifications.push(value); });
  return {
    metadata, history, outbox, cloudCalls, localCalls, notifications, accepted,
    get publisher() { return { ...publisher, retry: (id: string) => publisher.retry(resultId(id)) }; },
    save(id: string, state: ResultSharingRecord["state"] = "pending", actor_uid = actor, localId?: string) {
      const record = { ...execution(id, actor_uid), ...(localId ? { id: localId } : {}) };
      names.set(record.id, id);
      const item = { record, input_files: [{ id: "2".repeat(32), data_hash: "b".repeat(64) }] };
      history.set(id, structuredClone(item));
      metadata.set(id, { id: record.id, actor_uid, state, retryable: state !== "shared" && state !== "local_only" });
      if (state === "pending") outbox.set(id, structuredClone(item));
    },
    failCloud(error?: unknown) { cloudFailure = error; },
    acknowledge(value: unknown) { overrideAcknowledgement = true; acknowledgementOverride = value; },
    confirmAcknowledgement() { overrideAcknowledgement = false; },
    failAckOnce(error: unknown) { acknowledgementFailure = error; },
    failMetadata(error?: unknown) { metadataFailure = error; },
    failStatusWrite(error?: unknown) { statusWriteFailure = error; },
    failDenialPersistence(error?: unknown) { denialPersistenceFailure = error; },
    limit(full: boolean, large = false) { queueFull = full; oversized = large; },
    revoke() { denied = true; }, viewer() { role = "viewer"; },
    close() { closed = true; publisher.cancel(); },
    reopen() { denied = false; closed = false; publisher = createResultSharing(dependencies);
      publisher.subscribe((value) => { notifications.push(value); }); },
    delayCloud() { let release!: () => void; cloudWait = new Promise<void>((resolve) => { release = resolve; }); return release; },
    delayMetadata() { let release!: () => void; metadataWait = new Promise<void>((resolve) => { release = resolve; }); return release; },
  };
}

test("sharing snapshots contain only current-account metadata and recalculate counts", async () => {
  const f = fixture(); f.save("mine"); f.save("theirs", "failed", "account-b");
  const value = await f.publisher.refresh();
  assert.deepEqual(value, { records: [{ id: resultId("mine"), actor_uid: actor, state: "pending", retryable: true }], pending_count: 1, failed_count: 0 });
  assert.equal(JSON.stringify(value).includes("code"), false);
});

for (const status of [400, 409, 413, 422]) test(`permanent ${status} skips a poison head and publishes the following result`, async () => {
  const f = fixture(); f.save("poison"); f.save("valid");
  f.failCloud(new ApiError("Bad result", status, "INVALID_RESULT"));
  // Change the fixture response after the first request, without altering the
  // immutable result body or executing code again.
  f.publisher.subscribe((value) => { if (value.records.some((item) => item.id === resultId("poison") && item.state === "failed")) f.failCloud(); });
  f.publisher.start();
  await eventually(() => f.metadata.get("valid")?.state === "shared");
  assert.equal(f.metadata.get("poison")?.state, "failed");
  assert.equal(f.outbox.has("poison"), true);
  f.publisher.start(); await tick(); await tick();
  assert.deepEqual(f.cloudCalls.map((item) => item.id), ["poison", "valid"]);
});

test("offline delivery is durable across reopening and retries the original envelope without computing", async () => {
  const f = fixture(); f.save("offline");
  const original = structuredClone(f.history.get("offline"));
  f.failCloud(new TypeError("Network unavailable")); f.publisher.start();
  await eventually(() => Boolean(f.metadata.get("offline")?.error));
  assert.equal(f.metadata.get("offline")?.state, "pending");
  assert.equal(f.outbox.size, 1);
  f.close(); f.reopen(); f.failCloud(); f.publisher.start();
  await eventually(() => f.outbox.size === 0);
  assert.equal(f.metadata.get("offline")?.state, "shared");
  assert.equal(f.cloudCalls[0].body, f.cloudCalls[1].body);
  assert.deepEqual(f.history.get("offline"), original);
  assert.equal(f.localCalls.some((item) => item.path.includes("execute")), false);
});

test("accepted POST with failed local ack replays an identical idempotent body, never computation", async () => {
  const f = fixture(); f.save("run"); f.failAckOnce(new TypeError("Lost local response"));
  f.publisher.start(); await eventually(() => Boolean(f.metadata.get("run")?.error));
  assert.equal(f.accepted.size, 1); assert.equal(f.outbox.size, 1);
  await f.publisher.retry("run"); await eventually(() => f.outbox.size === 0);
  assert.equal(f.accepted.size, 1); assert.equal(f.cloudCalls.length, 2);
  assert.equal(f.cloudCalls[0].body, f.cloudCalls[1].body);
  assert.equal(f.localCalls.some((item) => item.path.includes("execute")), false);
});

test("post-acknowledgement status drains a stale pending read and notifies the durable shared state", async () => {
  const f = fixture(); f.save("run"); const releasePost = f.delayCloud(); f.publisher.start();
  await eventually(() => f.cloudCalls.length === 1);
  const releaseRead = f.delayMetadata(); const staleRead = f.publisher.refresh();
  releasePost(); await eventually(() => f.metadata.get("run")?.state === "shared");
  await tick(); await tick();
  assert.equal(f.localCalls.filter((call) => call.path === "/desktop-sharing").length, 2);
  releaseRead(); await staleRead;
  await eventually(() => f.notifications.at(-1)?.records[0].state === "shared");
  await tick(); await tick();
  assert.equal(f.localCalls.filter((call) => call.path === "/desktop-sharing").length, 3);
  assert.equal(f.notifications.at(-1)?.pending_count, 0);
});

test("concurrent forced reads coalesce one bounded fresh read after the prior loader", async () => {
  const f = fixture(); f.save("run"); const release = f.delayMetadata();
  const old = f.publisher.refresh();
  f.metadata.set("run", { ...f.metadata.get("run")!, state: "shared", retryable: false });
  const first = f.publisher.refresh(true), second = f.publisher.refresh(true);
  release(); await old;
  assert.equal((await first).records[0].state, "shared");
  assert.equal((await second).records[0].state, "shared");
  assert.equal(f.localCalls.filter((call) => call.path === "/desktop-sharing").length, 2);
});

for (const guard of ["revoked", "closed"]) test(`forced fresh metadata reads preserve the ${guard} guard after waiting`, async () => {
  const f = fixture(); f.save("run"); const release = f.delayMetadata();
  const old = f.publisher.refresh(), forced = f.publisher.refresh(true);
  const results = Promise.allSettled([old, forced]);
  if (guard === "revoked") f.revoke(); else f.close();
  release();
  const settled = await results;
  assert.ok(settled.every((result) => result.status === "rejected" &&
    (guard === "revoked" ? result.reason instanceof ApiError && result.reason.status === 403
      : result.reason instanceof DOMException && result.reason.name === "AbortError")));
  assert.equal(f.localCalls.filter((call) => call.path === "/desktop-sharing").length, 1);
  if (guard === "closed") assert.equal(f.notifications.length, 0);
});

for (const acknowledgement of [null, {}, { shared: false, id: cloudId(resultId("run")) },
  { shared: true }, { shared: true, id: "invalid" }, { shared: true, id: cloudId(resultId("another")) },
  { shared: true, id: resultId("run") }, { shared: true, id: cloudId(resultId("run"), "wrong-account") }])
  test(`unconfirmed publication ${JSON.stringify(acknowledgement)} retains the queue and surfaces pending status`, async () => {
    const f = fixture(); f.save("run"); f.acknowledge(acknowledgement); f.publisher.start();
    await eventually(() => f.metadata.get("run")?.error?.code === "SHARING_ACK_INVALID");
    assert.equal(f.metadata.get("run")?.state, "pending"); assert.equal(f.outbox.size, 1);
    assert.equal(f.localCalls.some((item) => item.path.startsWith("/desktop-outbox/")), false);
    f.confirmAcknowledgement(); await f.publisher.retry("run");
    await eventually(() => f.outbox.size === 0);
    assert.equal(f.cloudCalls[0].body, f.cloudCalls[1].body);
    assert.equal(f.localCalls.some((item) => item.path.includes("execute")), false);
  });

test("delivery error fields satisfy the bounded backend contract without masking the original failure", async () => {
  const f = fixture(); f.save("run");
  f.failCloud(new ApiError("Could not connect", -100, "invalid code!")); f.publisher.start();
  await eventually(() => Boolean(f.metadata.get("run")?.error));
  assert.deepEqual(f.metadata.get("run")?.error, { code: "RESULT_SHARING_FAILED", message: "Could not connect", status: 0 });
  f.failCloud(new ApiError("Still unavailable", 500, "A".repeat(100)));
  await f.publisher.retry("run");
  await eventually(() => f.metadata.get("run")?.error?.code === "A".repeat(64));
  assert.equal(f.metadata.get("run")?.error?.status, 500);
});

test("a crash-recovered shared queue item is acknowledged without a second publication", async () => {
  const f = fixture(); f.save("run");
  f.metadata.set("run", { ...f.metadata.get("run")!, state: "shared", retryable: false });
  f.publisher.start(); await eventually(() => f.outbox.size === 0);
  assert.equal(f.cloudCalls.length, 0);
});

for (const localId of ["aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa", "legacy-result-1"])
  test(`actual cloud archive acknowledges the actor-scoped SHA-256 ID for local result ${localId}`, async () => {
    const f = fixture(); f.save("real", "pending", actor, localId); f.publisher.start();
    await eventually(() => f.outbox.size === 0);
    const expected = createHash("sha256").update(`${actor}:${localId}`).digest("hex").slice(0, 32);
    assert.notEqual(expected, localId);
    assert.equal(f.accepted.has(expected), true);
    assert.equal(f.metadata.get("real")?.id, localId);
    assert.equal(f.metadata.get("real")?.state, "shared");
    assert.ok(f.localCalls.some((call) => call.path === `/desktop-outbox/${localId}?actor_uid=${actor}`));
  });

for (const large of [false, true]) test(`${large ? "oversized" : "full-queue"} retry failure remains visible and can be retried later`, async () => {
  const f = fixture(); f.save("recoverable", "failed"); f.limit(!large, large);
  await assert.rejects(f.publisher.retry("recoverable"), (error) => error instanceof ApiError && error.status === (large ? 413 : 409));
  const failed = f.notifications.at(-1)!;
  assert.equal(failed.failed_count, 1);
  assert.equal(failed.records[0].error?.code, large ? "OUTBOX_LIMIT" : "OUTBOX_FULL");
  assert.equal(f.cloudCalls.length, 0);
  f.limit(false); await f.publisher.retry("recoverable");
  await eventually(() => f.metadata.get("recoverable")?.state === "shared");
  assert.equal(f.localCalls.some((item) => item.path.includes("execute")), false);
});

test("a new result queued during an in-flight publication gets one subsequent bounded pass", async () => {
  const f = fixture(); f.save("first"); const release = f.delayCloud(); f.publisher.start();
  await eventually(() => f.cloudCalls.length === 1);
  f.save("second"); f.publisher.start(); release();
  await eventually(() => f.outbox.size === 0);
  assert.deepEqual(f.cloudCalls.map((item) => item.id), ["first", "second"]);
});

test("cloud authorization denial persists failure and stops later publications until a verified reopen", async () => {
  const f = fixture(); f.save("first"); f.save("second");
  f.failCloud(new ApiError("Membership removed", 403, "ROLE_REQUIRED")); f.publisher.start();
  await eventually(() => f.notifications.at(-1)?.error?.status === 403);
  assert.deepEqual(f.cloudCalls.map((item) => item.id), ["first"]);
  assert.equal(f.metadata.get("first")?.state, "failed");
  await assert.rejects(f.publisher.retry("first"), (error) => error instanceof ApiError && error.status === 403);
  f.reopen(); f.failCloud(); await f.publisher.retry("first");
  await eventually(() => f.outbox.size === 0);
});

for (const status of [401, 403]) for (const failedWrite of ["metadata", "denial", "both"])
  test(`cloud ${status} remains authoritative when ${failedWrite} local persistence fails`, async () => {
    const f = fixture(); f.save("first"); f.save("second");
    if (failedWrite !== "denial") f.failStatusWrite(new ApiError("Cannot write local status", 503, "LOCAL_STORAGE"));
    if (failedWrite !== "metadata") f.failDenialPersistence(new ApiError("Cannot write access state", 503, "LOCAL_STORAGE"));
    // Simulate losing metadata reads as soon as the cloud denies access. Those
    // reads cannot be used as a prerequisite for raising the access guard.
    f.publisher.subscribe((value) => {
      if (value.error?.status === status) f.failMetadata(new ApiError("Cannot read local status", 503, "LOCAL_STORAGE"));
    });
    f.failCloud(new ApiError("Cloud access removed", status, "MEMBERSHIP_REMOVED")); f.publisher.start();
    await eventually(() => f.notifications.at(-1)?.error?.code === "MEMBERSHIP_REMOVED");
    await tick(); await tick();
    assert.deepEqual(f.cloudCalls.map((item) => item.id), ["first"]);
    assert.equal(f.outbox.size, 2);
    const writes = f.localCalls.length;
    await assert.rejects(f.publisher.refresh(), (error) => error instanceof ApiError && error.status === 403);
    await assert.rejects(f.publisher.retry("first"), (error) => error instanceof ApiError && error.status === 403);
    f.publisher.start(); await tick(); await tick();
    f.publisher.report(new ApiError("Later disk failure", 503, "LOCAL_STORAGE"));
    assert.equal(f.localCalls.length, writes); // No later protected read/edit/ack.
    assert.deepEqual(f.notifications.at(-1)?.error, { code: "MEMBERSHIP_REMOVED", message: "Cloud access removed", status });
    assert.deepEqual(f.cloudCalls.map((item) => item.id), ["first"]);
  });

test("membership revoked during a cloud await cannot acknowledge or publish another result", async () => {
  const f = fixture(); f.save("first"); f.save("second");
  const release = f.delayCloud(); f.publisher.start();
  await eventually(() => f.cloudCalls.length === 1); f.revoke(); release();
  await eventually(() => f.notifications.at(-1)?.error?.status === 403);
  assert.equal(f.outbox.size, 2);
  assert.equal(f.localCalls.some((item) => item.path.startsWith("/desktop-outbox/")), false);
  assert.deepEqual(f.cloudCalls.map((item) => item.id), ["first"]);
});

test("local-only and other-actor results are never sent; viewers cannot retry", async () => {
  const f = fixture(); f.save("local", "local_only"); f.save("other", "pending", "account-b");
  f.publisher.start(); await tick(); await tick();
  assert.equal(f.cloudCalls.length, 0); assert.equal(f.outbox.size, 1);
  await assert.rejects(f.publisher.retry("local"), (error) => error instanceof ApiError && error.status === 403);
  await assert.rejects(f.publisher.retry("other"), (error) => error instanceof ApiError && error.status === 403);
  f.viewer(); await assert.rejects(f.publisher.retry("other"), (error) => error instanceof ApiError && error.status === 403);
  assert.equal(f.cloudCalls.length, 0);
});

test("closing a workspace during delivery suppresses callbacks and acknowledgement", async () => {
  const f = fixture(); f.save("first"); f.save("second"); const release = f.delayCloud();
  f.publisher.start(); await eventually(() => f.cloudCalls.length === 1);
  f.close(); const before = f.notifications.length; release(); await tick(); await tick();
  assert.equal(f.notifications.length, before);
  assert.equal(f.outbox.size, 2); assert.equal(f.cloudCalls.length, 1);
});

test("a metadata-read failure is surfaced without downloading full result bodies", async () => {
  const f = fixture(); f.failMetadata(new ApiError("Local metadata unavailable", 503, "LOCAL_STORAGE"));
  await assert.rejects(f.publisher.refresh());
  assert.equal(f.notifications.at(-1)?.error?.code, "LOCAL_STORAGE");
  assert.deepEqual(f.localCalls.map((item) => item.path), ["/desktop-sharing"]);
});

test("cloud archive drops private delivery context while retaining ordered events and LaTeX", () => {
  const value = { ...execution("private"), sharing_context: { actor_uid: actor, input_files: [] },
    sharing: { state: "pending" }, local_only: false,
    sharing_state: "pending", sharing_error: { message: "local failure" }, sharing_retryable: true,
    outputs: [{ type: "text" as const, data: "42", latex: "\\begin{tabular}{r}42\\end{tabular}" }] };
  const original = structuredClone(value);
  const archived = cloudArchiveRecord(value).record;
  assert.equal("sharing_context" in archived, false); assert.equal("sharing_state" in archived, false);
  assert.equal("sharing_error" in archived, false); assert.equal("sharing_retryable" in archived, false);
  assert.equal("sharing" in archived, false); assert.equal("local_only" in archived, false);
  assert.deepEqual(archived.events, value.events); assert.deepEqual(archived.outputs, value.outputs);
  assert.deepEqual(value, original);
});

test("unavailable WebCrypto keeps delivery pending instead of accepting an unverifiable acknowledgement", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto")!;
  const f = fixture(); f.save("run");
  try {
    Object.defineProperty(globalThis, "crypto", { value: undefined, configurable: true });
    f.publisher.start();
    await eventually(() => f.metadata.get("run")?.error?.code === "SHARING_ACK_UNVERIFIED");
    assert.equal(f.cloudCalls.length, 0); assert.equal(f.outbox.size, 1);
    assert.equal(f.metadata.get("run")?.state, "pending");
  } finally {
    Object.defineProperty(globalThis, "crypto", descriptor);
  }
  await f.publisher.retry("run"); await eventually(() => f.outbox.size === 0);
});
