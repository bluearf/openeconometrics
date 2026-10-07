/** Delivery metadata is independent of computation and of the result payload. */
import { ApiError, type WorkspaceClient } from "./api.ts";
import type { ExecutionRecord } from "./types.ts";

export interface ResultSharingError {
  code: string;
  message: string;
  status?: number;
}
export interface ResultSharingRecord {
  id: string;
  actor_uid: string;
  state: "pending" | "shared" | "failed" | "local_only";
  error?: ResultSharingError;
  retryable: boolean;
}
export interface ResultSharingSnapshot {
  records: ResultSharingRecord[];
  pending_count: number;
  failed_count: number;
  error?: ResultSharingError;
}
export interface ResultOutboxItem {
  record: ExecutionRecord;
  input_files: { id: string; data_hash: string }[];
}
interface Dependencies {
  actorUid: string;
  local: WorkspaceClient;
  cloud<T>(path: string, options?: RequestInit): Promise<T>;
  accessible(editing?: boolean): unknown;
  closed(): boolean;
  deny(): Promise<void>;
  archive(record: ExecutionRecord): { record: ExecutionRecord };
}

const empty = (): ResultSharingSnapshot => ({ records: [], pending_count: 0, failed_count: 0 });
function failure(error: unknown, fallback = "RESULT_SHARING_FAILED"): ResultSharingError {
  const candidate = error instanceof ApiError ? error.code : "";
  const status = error instanceof ApiError ? error.status : 0;
  return {
    code: candidate && /^[A-Za-z0-9_-]+$/.test(candidate) ? candidate.slice(0, 64) : fallback,
    message: (error instanceof Error ? error.message : "Could not share this result.").slice(0, 256),
    status: status === 0 || Number.isInteger(status) && status >= 100 && status <= 599 ? status : 0,
  };
}
function aborted(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}
async function remoteResultId(actorUid: string, clientId: string): Promise<string> {
  if (!globalThis.crypto?.subtle)
    throw new ApiError("Could not verify the cloud sharing acknowledgement. Retry sharing in the desktop app.", 0, "SHARING_ACK_UNVERIFIED");
  // The cloud archive scopes local execution identifiers (normally UUIDs) to
  // the authenticated actor; its archive ID is not the local execution ID.
  const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${actorUid}:${clientId}`));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("").slice(0, 32);
}

/** A single bounded publisher. Retrying delivery never executes Python. */
export function createResultSharing(deps: Dependencies) {
  let snapshot = empty();
  let loading: Promise<ResultSharingSnapshot> | null = null;
  let publishing: Promise<void> | null = null;
  let restartRequested = false;
  const listeners = new Set<(value: ResultSharingSnapshot) => void>();

  function notify() {
    if (deps.closed()) return;
    for (const listener of listeners) {
      try { listener(structuredClone(snapshot)); }
      catch { /* A detached view cannot interrupt durable delivery. */ }
    }
  }
  function report(error: unknown, authoritative = false) {
    if (deps.closed() || aborted(error)) return;
    // Once access has been denied, a later local storage/network error must
    // not hide that guard. A verified reopen refreshes the snapshot normally.
    if (!authoritative && [401, 403].includes(snapshot.error?.status ?? 0)) return;
    snapshot = { ...snapshot, error: failure(error) };
    notify();
  }
  async function refresh(afterMutation = false): Promise<ResultSharingSnapshot> {
    deps.accessible();
    if (loading) {
      if (!afterMutation) return loading;
      // An in-flight read may have captured metadata before the completed
      // write. Drain that one loader, then coalesce one fresh read. Never join
      // the stale response as a mutation's final status or loop on failures.
      try { await loading; }
      catch { /* The loader reports its own failure. */ }
      deps.accessible();
      return refresh();
    }
    loading = (async () => {
      const value = await deps.local.request<ResultSharingSnapshot>("/desktop-sharing");
      deps.accessible();
      if (!value || !Array.isArray(value.records) || value.records.length > 520)
        throw new Error("Could not verify result sharing status.");
      const records = value.records.filter((record) => record.actor_uid === deps.actorUid).map((record) => {
        if (typeof record.id !== "string" || !["pending", "shared", "failed", "local_only"].includes(record.state)
            || typeof record.retryable !== "boolean") throw new Error("Could not verify result sharing status.");
        // Copy only metadata, even if a backend accidentally returns result bodies.
        return { id: record.id, actor_uid: record.actor_uid, state: record.state,
          retryable: record.retryable, ...(record.error ? { error: failure(new ApiError(
            record.error.message, record.error.status ?? 0, record.error.code)) } : {}) };
      });
      snapshot = { records, pending_count: records.filter((record) => record.state === "pending").length,
        failed_count: records.filter((record) => record.state === "failed").length };
      notify();
      return structuredClone(snapshot);
    })().catch((error) => { report(error); throw error; }).finally(() => { loading = null; });
    return loading;
  }
  async function update(id: string, state: "pending" | "failed", error: unknown) {
    deps.accessible(true);
    await deps.local.request(`/desktop-sharing/${encodeURIComponent(id)}`, { method: "PUT",
      body: JSON.stringify({ actor_uid: deps.actorUid, state, error: failure(error) }) });
    deps.accessible(true);
  }
  async function flush() {
    if (publishing) return publishing;
    publishing = (async () => {
      deps.accessible(true);
      const current = await refresh(true);
      deps.accessible(true);
      const outbox = await deps.local.request<{ items: ResultOutboxItem[] }>("/desktop-outbox");
      deps.accessible(true);
      if (!outbox || !Array.isArray(outbox.items) || outbox.items.length > 20)
        throw new Error("Could not verify the result sharing queue.");
      const records = new Map(current.records.map((record) => [record.id, record]));
      for (const item of outbox.items) {
        deps.accessible(true);
        if (item.record.actor_uid !== deps.actorUid || item.record.local_only === true) continue;
        const metadata = records.get(item.record.id);
        if (metadata?.state === "failed" || metadata?.state === "local_only") continue;
        if (metadata?.state !== "shared") {
          const archive = deps.archive(item.record);
          try {
            const expectedId = await remoteResultId(deps.actorUid, item.record.id);
            deps.accessible(true);
            const acknowledgement = await deps.cloud<{ shared: boolean; id: string }>("/desktop/results", { method: "POST",
              body: JSON.stringify({ record: archive.record, input_files: item.input_files }) });
            deps.accessible(true);
            if (acknowledgement?.shared !== true || typeof acknowledgement.id !== "string"
                || !/^[0-9a-f]{32}$/.test(acknowledgement.id) || acknowledgement.id !== expectedId)
              throw new ApiError("Could not confirm that the result was shared. Retry sharing.", 0, "SHARING_ACK_INVALID");
          } catch (error) {
            if (deps.closed() || aborted(error)) throw error;
            // An independently revoked workspace must not be sent further requests.
            deps.accessible(true);
            const denied = error instanceof ApiError && (error.status === 401 || error.status === 403);
            if (denied) {
              // Raise the in-memory guard immediately. Its disk write and the
              // optional result-status write must not delay access revocation
              // or replace the authoritative cloud error with a storage error.
              let denial = Promise.resolve();
              try { denial = deps.deny().catch(() => {}); }
              catch { /* deny raises its guard before attempting persistence. */ }
              report(error, true);
              try {
                await deps.local.request(`/desktop-sharing/${encodeURIComponent(item.record.id)}`, { method: "PUT",
                  body: JSON.stringify({ actor_uid: deps.actorUid, state: "failed", error: failure(error) }) });
                if (!deps.closed()) {
                  snapshot = { ...snapshot, records: snapshot.records.map((record) => record.id === item.record.id
                    ? { ...record, state: "failed", error: failure(error), retryable: true } : record) };
                  snapshot.pending_count = snapshot.records.filter((record) => record.state === "pending").length;
                  snapshot.failed_count = snapshot.records.filter((record) => record.state === "failed").length;
                }
              } catch { /* Local metadata is best effort after permission is lost. */ }
              await denial;
              report(error, true);
              return;
            }
            const permanent = error instanceof ApiError && [400, 409, 413, 422].includes(error.status);
            await update(item.record.id, permanent ? "failed" : "pending", error);
            await refresh(true);
            if (permanent) continue;
            return; // Connectivity retries only on a later online event or manual retry.
          }
        }
        try {
          deps.accessible(true);
          await deps.local.request(`/desktop-outbox/${encodeURIComponent(item.record.id)}?actor_uid=${encodeURIComponent(deps.actorUid)}`, { method: "DELETE" });
          deps.accessible(true);
          await refresh(true);
        } catch (error) {
          if (deps.closed() || aborted(error)) throw error;
          deps.accessible(true);
          // POST succeeded but local acknowledgement failed. Keep the immutable
          // queue item; the server accepts the identical publication idempotently.
          try { await update(item.record.id, "pending", error); await refresh(true); }
          catch (persistenceError) { report(persistenceError); }
          return;
        }
      }
    })().catch((error) => { report(error); throw error; }).finally(() => { publishing = null; });
    return publishing;
  }
  function start() {
    if (deps.closed()) return;
    if (publishing) { restartRequested = true; return; }
    void flush().catch(() => { /* flush reports the guarded status to subscribers. */ }).finally(() => {
      if (restartRequested && !deps.closed()) {
        restartRequested = false;
        start();
      }
    });
  }
  async function retry(id: string) {
    deps.accessible(true);
    try {
      await deps.local.request(`/desktop-sharing/${encodeURIComponent(id)}/retry`, { method: "POST",
        body: JSON.stringify({ actor_uid: deps.actorUid }) });
    } catch (error) {
      try { await refresh(true); }
      catch { /* refresh already reports its own failure. */ }
      report(error);
      throw error;
    }
    deps.accessible(true);
    const value = await refresh(true);
    start();
    return value;
  }
  return {
    refresh, start, retry, report,
    subscribe(listener: (value: ResultSharingSnapshot) => void) {
      if (deps.closed()) return () => {};
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    cancel() { restartRequested = false; listeners.clear(); },
  };
}
