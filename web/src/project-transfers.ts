/** Visible, explicit native sharing retries. No credential enters a UI snapshot. */
import { ApiError } from "./api.ts";
import type { DatasetProfile } from "./types.ts";

export interface ProjectTransfer {
  request_id: string;
  name: string;
  size_bytes: number;
  sha256: string;
  state: "waiting" | "uploading" | "ready" | "cancel_pending";
  acknowledged_bytes: number;
}
export interface TransferCatalogueFile {
  id: string;
  name: string;
  data_hash: string;
  size_bytes: number;
  transfer: "chunked-v1";
}
export interface ProjectTransferSnapshot {
  transfers: readonly ProjectTransfer[];
  busyRequestId: string | null;
  error: string;
}
export interface ProjectTransferActions {
  getSnapshot(): ProjectTransferSnapshot;
  subscribe(listener: (value: ProjectTransferSnapshot) => void): () => void;
  refresh(): Promise<void>;
  status(requestId: string): Promise<void>;
  resume(requestId: string): Promise<DatasetProfile | null>;
  cancel(requestId: string): Promise<void>;
  dispose(): void;
}
export interface ProjectTransferDependencies {
  invoke(
    action: "list" | "status" | "resume" | "cancel" | "cancel_download",
    requestId?: string,
  ): Promise<{ status: number; body: unknown }>;
  assertAccess(editing: boolean): void;
  importReady(
    file: TransferCatalogueFile,
    assertCurrent?: () => void,
  ): Promise<DatasetProfile>;
}
const MAX_FILE_BYTES = 2 * 1024 ** 3;
const ID = /^[0-9a-f]{32}$/;
const HASH = /^[0-9a-f]{64}$/;
function invalid() {
  return new ApiError(
    "The file transfer could not be verified.",
    0,
    "DATA_INTEGRITY",
  );
}
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw invalid();
  return value as Record<string, unknown>;
}
export function validateProjectTransfer(value: unknown): ProjectTransfer {
  const item = object(value);
  if (
    Object.keys(item).some(
      (key) =>
        ![
          "request_id",
          "name",
          "size_bytes",
          "sha256",
          "state",
          "acknowledged_bytes",
        ].includes(key),
    ) ||
    typeof item.request_id !== "string" ||
    !ID.test(item.request_id) ||
    typeof item.name !== "string" ||
    !item.name ||
    item.name.length > 180 ||
    /[\\/\x00-\x1f\x7f]/.test(item.name) ||
    typeof item.sha256 !== "string" ||
    !HASH.test(item.sha256) ||
    !Number.isSafeInteger(item.size_bytes) ||
    Number(item.size_bytes) < 1 ||
    Number(item.size_bytes) > MAX_FILE_BYTES ||
    !Number.isSafeInteger(item.acknowledged_bytes) ||
    Number(item.acknowledged_bytes) < 0 ||
    Number(item.acknowledged_bytes) > Number(item.size_bytes) ||
    !["waiting", "uploading", "ready", "cancel_pending"].includes(
      String(item.state),
    )
  )
    throw invalid();
  return {
    request_id: item.request_id,
    name: item.name,
    size_bytes: Number(item.size_bytes),
    sha256: item.sha256,
    state: item.state as ProjectTransfer["state"],
    acknowledged_bytes: Number(item.acknowledged_bytes),
  };
}
function sameSource(next: ProjectTransfer, previous: ProjectTransfer) {
  if (
    next.request_id !== previous.request_id ||
    next.name !== previous.name ||
    next.size_bytes !== previous.size_bytes ||
    next.sha256 !== previous.sha256
  )
    throw invalid();
}
function readyFile(
  value: unknown,
  pin: ProjectTransfer,
): TransferCatalogueFile {
  const file = object(value);
  if (
    typeof file.id !== "string" ||
    !ID.test(file.id) ||
    file.name !== pin.name ||
    file.size_bytes !== pin.size_bytes ||
    file.data_hash !== pin.sha256 ||
    file.transfer !== "chunked-v1" ||
    "blob" in file ||
    "token" in file ||
    "local_path" in file
  )
    throw invalid();
  return {
    id: file.id,
    name: pin.name,
    size_bytes: pin.size_bytes,
    data_hash: pin.sha256,
    transfer: "chunked-v1",
  };
}
export function transferErrorMessage(error: unknown): string {
  if (
    error instanceof ApiError &&
    error.status === 409 &&
    error.code === "TRANSFER_COMPLETE"
  )
    return "Sharing is already complete. Resume to open the shared file; cancellation cannot remove a published file.";
  if (error instanceof ApiError && error.code === "CLEANUP_PENDING")
    return "Cancellation is pending. Keep the local file and retry cleanup when connected.";
  if (
    error instanceof ApiError &&
    (error.status === 401 || error.status === 403)
  )
    return "Could not verify permission to share this project's data. Reopen the project.";
  if (
    error instanceof TypeError ||
    (error instanceof ApiError && error.status >= 500)
  )
    return "Connection lost. The local file is available; sharing can be resumed.";
  if (
    (error instanceof ApiError && error.code === "DATA_INTEGRITY") ||
    error === "OPENECON_DATA_INTEGRITY"
  )
    return "The file could not be verified. Keep the local file and check the pending transfer.";
  return "Could not finish sharing. Keep the local file and retry or cancel the pending transfer.";
}

export function createProjectTransferActions(
  deps: ProjectTransferDependencies,
): ProjectTransferActions {
  let snapshot: ProjectTransferSnapshot = {
    transfers: [],
    busyRequestId: null,
    error: "",
  };
  let closed = false,
    refreshSequence = 0;
  const generations = new Map<string, number>();
  const cancelling = new Set<string>();
  const listeners = new Set<(value: ProjectTransferSnapshot) => void>();
  const copy = () => structuredClone(snapshot);
  const publish = (change: Partial<ProjectTransferSnapshot>) => {
    if (closed) return;
    snapshot = { ...snapshot, ...change };
    for (const listener of listeners) {
      try {
        listener(copy());
      } catch {
        /* A detached panel cannot fail a durable transfer. */
      }
    }
  };
  const active = (editing: boolean) => {
    if (closed) throw new DOMException("Project closed", "AbortError");
    deps.assertAccess(editing);
  };
  const pin = (id: string) => {
    if (!ID.test(id)) throw invalid();
    const found = snapshot.transfers.find((item) => item.request_id === id);
    if (!found) throw invalid();
    return found;
  };
  const update = (view: ProjectTransfer) =>
    publish({
      transfers: snapshot.transfers.map((item) =>
        item.request_id === view.request_id
          ? {
              ...view,
              ...(item.state === "cancel_pending"
                ? { state: "cancel_pending" as const }
                : {}),
            }
          : item,
      ),
    });
  const response = async (
    action: "list" | "status" | "resume" | "cancel",
    id?: string,
  ) => {
    active(action === "resume" || action === "cancel");
    const result = await deps.invoke(action, id);
    active(action === "resume" || action === "cancel");
    if (
      !Number.isInteger(result.status) ||
      result.status < 200 ||
      result.status >= 300
    ) {
      const detail =
        result.body && typeof result.body === "object"
          ? (result.body as { detail?: unknown }).detail
          : null;
      const code =
        detail && typeof detail === "object"
          ? (detail as { code?: unknown }).code
          : "";
      throw new ApiError(
        "Could not complete file sharing.",
        result.status,
        typeof code === "string" && /^[A-Z_]{1,64}$/.test(code) ? code : "",
      );
    }
    return result;
  };
  const report = (error: unknown) => {
    if (
      !closed &&
      !(error instanceof DOMException && error.name === "AbortError")
    )
      publish({ error: transferErrorMessage(error) });
  };
  const controller: ProjectTransferActions = {
    getSnapshot: copy,
    subscribe(listener) {
      listeners.add(listener);
      listener(copy());
      return () => {
        listeners.delete(listener);
      };
    },
    async refresh() {
      const sequence = ++refreshSequence;
      try {
        const result = await response("list");
        const body = object(result.body);
        if (!Array.isArray(body.transfers) || body.transfers.length > 4)
          throw invalid();
        const views = body.transfers.map(validateProjectTransfer);
        if (new Set(views.map((item) => item.request_id)).size !== views.length)
          throw invalid();
        for (const view of views) {
          const previous = snapshot.transfers.find(
            (item) => item.request_id === view.request_id,
          );
          if (previous) sameSource(view, previous);
        }
        if (sequence === refreshSequence)
          publish({
            transfers: views.map((view) =>
              snapshot.transfers.some(
                (item) =>
                  item.request_id === view.request_id &&
                  item.state === "cancel_pending",
              )
                ? { ...view, state: "cancel_pending" }
                : view,
            ),
            error: "",
          });
      } catch (error) {
        if (sequence === refreshSequence) report(error);
        throw error;
      }
    },
    async status(id) {
      const previous = pin(id),
        generation = generations.get(id) || 0;
      try {
        const result = await response("status", id);
        const view = validateProjectTransfer(result.body);
        sameSource(view, previous);
        if ((generations.get(id) || 0) === generation) update(view);
      } catch (error) {
        if ((generations.get(id) || 0) === generation) report(error);
        throw error;
      }
    },
    async resume(id) {
      active(true);
      const previous = pin(id);
      if (snapshot.busyRequestId || previous.state === "cancel_pending")
        throw new Error("Finish or cancel the current transfer first.");
      const generation = (generations.get(id) || 0) + 1;
      generations.set(id, generation);
      ++refreshSequence;
      publish({ busyRequestId: id, error: "" });
      try {
        const result = await response("resume", id);
        if (generations.get(id) !== generation) return null;
        const body = object(result.body);
        if (result.status === 202) {
          const view = validateProjectTransfer(body);
          sameSource(view, previous);
          update(view);
          return null;
        }
        if (result.status !== 200 || body.state !== "ready") throw invalid();
        const file = readyFile(body.file, previous);
        const imported = await deps.importReady(file, () => {
          active(true);
          if (generations.get(id) !== generation)
            throw new DOMException("Transfer cancelled", "AbortError");
        });
        active(true);
        if (generations.get(id) !== generation) return null;
        const cached = imported as DatasetProfile & {
          cloud_id?: string;
          sha256?: string;
        };
        if (
          cached.cloud_id !== file.id ||
          cached.sha256 !== file.data_hash ||
          !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(
            imported.id,
          ) ||
          imported.name !== file.name
        )
          throw invalid();
        // A list read begun during this operation can still contain its journal.
        ++refreshSequence;
        publish({
          transfers: snapshot.transfers.filter(
            (item) => item.request_id !== id,
          ),
          error: "",
        });
        return imported;
      } catch (error) {
        if (generations.get(id) === generation) report(error);
        throw error;
      } finally {
        if (snapshot.busyRequestId === id) publish({ busyRequestId: null });
      }
    },
    async cancel(id) {
      active(true);
      if (cancelling.has(id)) return;
      const previous = pin(id);
      cancelling.add(id);
      generations.set(id, (generations.get(id) || 0) + 1);
      ++refreshSequence;
      update({ ...previous, state: "cancel_pending" });
      publish({ error: "" });
      try {
        const result = await response("cancel", id);
        const body = object(result.body);
        if (result.status === 202) {
          const view = validateProjectTransfer(body);
          sameSource(view, previous);
          if (view.state !== "cancel_pending") throw invalid();
          update(view);
          return;
        }
        if (
          result.status !== 200 ||
          body.cancelled !== true ||
          body.request_id !== id
        )
          throw invalid();
        ++refreshSequence;
        publish({
          transfers: snapshot.transfers.filter(
            (item) => item.request_id !== id,
          ),
          error: "",
        });
      } catch (error) {
        if (
          error instanceof ApiError &&
          error.status === 409 &&
          error.code === "TRANSFER_COMPLETE"
        ) {
          ++refreshSequence;
          publish({
            transfers: snapshot.transfers.map((item) =>
              item.request_id === id ? { ...item, state: "ready" } : item,
            ),
          });
        }
        report(error);
        throw error;
      } finally {
        cancelling.delete(id);
      }
    },
    dispose() {
      closed = true;
      refreshSequence++;
      listeners.clear();
    },
  };
  return controller;
}
