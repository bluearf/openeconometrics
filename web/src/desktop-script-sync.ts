/** Named files retain independent local versions and acknowledged cloud bases. */
import { ApiError, scriptFilePath, validateScriptFile, type ScriptFile, type ScriptSummary, type WorkspaceClient } from "./api.ts";
import { documentExtension, validFileName } from "./file-layout.ts";
import { isTransientDesktopFailure } from "./desktop.ts";

export interface FileSyncState {
  name: string;
  cloud_version: number | null;
  local_version: number;
  // For an unacknowledged creation this is the immutable, idempotent POST body.
  base_code: string;
  conflict?: boolean;
}
export interface DesktopSyncState {
  cloud_version: number;
  base_code: string;
  role: "owner" | "editor" | "viewer";
  scripts?: Record<string, FileSyncState>;
  script_catalog?: ScriptSummary[];
  access_denied?: boolean;
}
export type SyncStateUpdate = DesktopSyncState | ((state: DesktopSyncState) => DesktopSyncState);
interface Dependencies {
  local: WorkspaceClient;
  cloud<T>(path: string, options?: RequestInit): Promise<T>;
  state(): DesktopSyncState | null;
  persist(state: SyncStateUpdate): Promise<void>;
  closed?(): boolean;
}
const conflict = () => new ApiError("The cloud version of this file has changed. Your local draft is preserved.", 409, "VERSION_CONFLICT");
const denied = () => new ApiError("Could not verify your access to this project. Reopen the project.", 403, "ROLE_REQUIRED");
const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) });

function document(value: ScriptFile, id: string): ScriptFile {
  if (!value || value.id !== id || typeof value.code !== "string" || Array.from(value.code).length > 64000
      || !Number.isSafeInteger(value.version) || value.version < 0)
    throw new Error("Could not verify the file.");
  return validateScriptFile(value, id);
}
function metadata(value: { scripts: ScriptSummary[] }): ScriptSummary[] {
  if (!value || !Array.isArray(value.scripts) || value.scripts.length > 200)
    throw new Error("Could not verify the file list.");
  const ids = new Set<string>();
  for (const item of value.scripts) {
    scriptFilePath(item.id);
    if (ids.has(item.id) || typeof item.name !== "string" || !documentExtension(item.name)
        || (item.id === "analysis" && documentExtension(item.name) !== ".py")
        || !Number.isSafeInteger(item.version) || item.version < 0)
      throw new Error("Could not verify the file list.");
    validFileName(item.name, "script");
    ids.add(item.id);
  }
  if (!ids.has("analysis")) throw new Error("The main Python file could not be found.");
  // Never persist accidental document bodies in the metadata cache.
  return value.scripts.map(({ id, name, version }) => ({ id, name, version }));
}

/** Network operations serialize externally; local CAS commits never wait for them. */
export function createNamedScriptSync(deps: Dependencies) {
  let localSerial: Promise<unknown> = Promise.resolve();
  function localOperation<T>(operation: () => Promise<T>): Promise<T> {
    const next = localSerial.catch(() => {}).then(() => {
      current();
      return operation();
    });
    localSerial = next;
    return next;
  }
  function current(): DesktopSyncState {
    if (deps.closed?.()) throw new DOMException("Workspace closed", "AbortError");
    const state = deps.state();
    if (!state) throw new Error("Open the project first.");
    if (state.access_denied) throw denied();
    return state;
  }
  function editable() {
    if (current().role === "viewer") throw denied();
  }
  async function cloud<T>(path: string, options?: RequestInit): Promise<T> {
    try { return await deps.cloud<T>(path, options); }
    catch (error) {
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        if (deps.state() && !deps.closed?.())
          await deps.persist((state) => ({ ...state, access_denied: true }));
      }
      throw error;
    }
  }
  async function remember(id: string, entry: FileSyncState) {
    current();
    await deps.persist((state) => ({ ...state, scripts: { ...state.scripts, [id]: entry } }));
  }
  function decorated(local: ScriptFile): ScriptFile {
    const entry = current().scripts?.[local.id];
    return { ...local, pending_sync: !entry || entry.cloud_version === null || local.code !== entry.base_code || Boolean(entry.conflict),
      ...(entry?.conflict ? { sync_conflict: true } : {}) };
  }
  async function localFile(id: string): Promise<ScriptFile | null> {
    try { return document(await deps.local.request<ScriptFile>(scriptFilePath(id)), id); }
    catch (error) {
      if (error instanceof ApiError && error.status === 404) return null;
      throw error;
    }
  }
  async function cache(remote: ScriptFile, local: ScriptFile | null): Promise<ScriptFile> {
    // The private local route uses the local version; a cloud version can never
    // bypass another writer's local optimistic check or silently rename a peer.
    return document(await deps.local.request<ScriptFile>(`/desktop-scripts/${remote.id}`,
      json("PUT", { name: remote.name, code: remote.code, version: local?.version ?? 0 })), remote.id);
  }
  async function acknowledge(remote: ScriptFile, local: ScriptFile) {
    await remember(remote.id, { name: remote.name, cloud_version: remote.version,
      base_code: remote.code!, local_version: local.code === remote.code ? local.version
        : (current().scripts?.[remote.id]?.local_version ?? 0) });
  }
  async function readLocal(id: string): Promise<ScriptFile | null> {
    return localOperation(async () => {
      const local = await localFile(id);
      // An untracked cache must first establish its exact remote base.
      return local && current().scripts?.[id] ? decorated(local) : null;
    });
  }
  async function catalog(): Promise<{ scripts: ScriptSummary[] }> {
    current();
    let remote = current().script_catalog ?? [];
    try {
      remote = metadata(await cloud<{ scripts: ScriptSummary[] }>("/console/scripts"));
      await deps.persist((state) => ({ ...state, script_catalog: remote }));
    } catch (error) {
      // An old deployment's 404 or an authorization denial is never "offline".
      if (!isTransientDesktopFailure(error)) throw error;
    }
    const local = metadata(await deps.local.request<{ scripts: ScriptSummary[] }>("/console/scripts"));
    const merged = new Map(remote.map((entry) => [entry.id, entry]));
    for (const entry of local) {
      const saved = current().scripts?.[entry.id];
      merged.set(entry.id, entry.id === "analysis"
        ? { ...entry }
        : { ...entry, pending_sync: !saved || saved.cloud_version === null || saved.local_version !== entry.version || Boolean(saved.conflict) });
    }
    if (merged.size > 200) throw new ApiError("The Python file limit has been reached. Your local drafts are preserved.", 409, "SCRIPT_LIMIT");
    return { scripts: [...merged.values()].sort((a, b) => a.id === "analysis" ? -1 : b.id === "analysis" ? 1 : 0) };
  }
  async function read(id: string, resolveConflict = false, sourceChanged?: () => void): Promise<ScriptFile> {
    current();
    const local = await localFile(id);
    const entry = current().scripts?.[id];
    let remote: ScriptFile;
    try { remote = document(await cloud<ScriptFile>(scriptFilePath(id)), id); }
    catch (error) {
      if (resolveConflict) throw error;
      if (local && isTransientDesktopFailure(error)) return decorated(local);
      if (local && entry?.cloud_version === null && error instanceof ApiError && error.status === 404) {
        // Distinguish an absent new document from a server lacking this API.
        await catalog();
        return decorated(local);
      }
      throw error;
    }
    return localOperation(async () => {
      // A local edit may have finished while the cloud response was in flight.
      const latest = await localFile(id);
      const saved = current().scripts?.[id];
      if (resolveConflict) {
        // Explicit reload is still CAS protected against another local writer.
        const copied = await cache(remote, local);
        await acknowledge(remote, copied);
        return decorated(copied);
      }
      if (latest && saved && !saved.conflict && latest.code !== saved.base_code && remote.code === saved.base_code) {
        await acknowledge(remote, latest);
        return decorated(latest);
      }
      if (latest && (saved?.conflict || (latest.code !== (saved?.base_code ?? remote.code)
          && remote.code !== latest.code && (!saved || remote.version !== saved.cloud_version)))) {
        await remember(id, { name: saved?.name ?? remote.name, cloud_version: saved?.cloud_version ?? remote.version,
          base_code: saved?.base_code ?? remote.code!, local_version: saved?.local_version ?? latest.version, conflict: true });
        return decorated(latest);
      }
      if (latest && saved && latest.code !== saved.base_code && remote.version === saved.cloud_version) return decorated(latest);
      const copied = latest && latest.name === remote.name && latest.code === remote.code
        ? latest : await cache(remote, latest);
      await acknowledge(remote, copied);
      if (latest && (latest.name !== copied.name || latest.code !== copied.code)) sourceChanged?.();
      return decorated(copied);
    });
  }
  async function synchronize(local: ScriptFile): Promise<ScriptFile> {
    editable();
    const initialEntry = current().scripts?.[local.id];
    if (!initialEntry) throw new Error("The file's sync record could not be found. Reopen the file.");
    let entry: FileSyncState = initialEntry;
    if (entry.conflict) throw conflict();
    try {
      if (entry.cloud_version === null) {
        const remote = document(await cloud<ScriptFile>("/console/scripts", json("POST", {
          id: local.id, name: entry.name, code: entry.base_code,
        })), local.id);
        if (remote.code !== entry.base_code) throw new Error("Could not verify the new file's cloud record.");
        await localOperation(async () => {
          const latest = await localFile(local.id);
          if (!latest) throw new Error("The local Python file could not be found.");
          if (latest.name !== remote.name)
            local = await cache({ ...latest, name: remote.name }, latest);
          await acknowledge(remote, local);
        });
        entry = current().scripts![local.id];
      }
      if (local.code !== entry.base_code) {
        const remote = document(await cloud<ScriptFile>(scriptFilePath(local.id), json("PUT", {
          code: local.code, version: entry.cloud_version,
        })), local.id);
        if (remote.code !== local.code || remote.name !== local.name)
          throw new Error("Could not verify the file's cloud record.");
        await localOperation(() => acknowledge(remote, local));
      } else if (entry.local_version !== local.version) {
        await localOperation(() => remember(local.id, { ...entry, local_version: local.version }));
      }
      return (await readLocal(local.id))!;
    } catch (error) {
      // Older cloud releases accept only .py names. The validated document and
      // its original POST body are already durable locally; retain the pending
      // receipt only for that exact legacy filename rejection, never an access,
      // concurrency or other validation error. A later retry still posts it.
      if (entry.cloud_version === null && [".md", ".tex"].includes(documentExtension(entry.name) ?? "")
          && error instanceof ApiError && error.status === 422 && error.code === "INVALID_SCRIPT"
          && ["Use a short, safe .py filename.", "Güvenli, kısa bir .py dosya adı kullanın."].includes(error.message))
        return (await readLocal(local.id))!;
      if (error instanceof ApiError && error.status === 409) {
        // An acknowledgement can be lost after a successful PUT. Exact content
        // readback settles it without overwriting a collaborator's newer text.
        const remote = document(await cloud<ScriptFile>(scriptFilePath(local.id)), local.id);
        if (entry.cloud_version !== null && remote.code === local.code && remote.name === local.name) {
          await localOperation(() => acknowledge(remote, local));
          return (await readLocal(local.id))!;
        }
        await localOperation(() => remember(local.id, { ...entry, conflict: true }));
        throw conflict();
      }
      if (!isTransientDesktopFailure(error)) throw error;
      return (await readLocal(local.id))!;
    }
  }
  async function create(body: string): Promise<ScriptFile> {
    editable();
    // Checking the catalog catches an older cloud deployment before claiming a
    // synchronized creation; transient failures still allow durable offline work.
    await catalog();
    const payload = JSON.parse(body) as Record<string, unknown>;
    if (!payload || typeof payload !== "object" || Array.isArray(payload)
        || Object.keys(payload).some((key) => !["id", "name", "code"].includes(key)))
      throw new Error("Invalid Python file.");
    const id = payload.id ?? crypto.randomUUID().replaceAll("-", "");
    if (typeof id !== "string" || !/^[0-9a-f]{32}$/.test(id)) throw new Error("Invalid file ID.");
    // A supplied existing identifier must never claim another file's sync state.
    if ((current().script_catalog ?? []).some((file) => file.id === id) || await localFile(id))
      throw new ApiError("This file ID is already in use.", 409, "SCRIPT_ID_CONFLICT");
    const local = await localOperation(async () => {
      editable();
      const created = document(await deps.local.request<ScriptFile>("/console/scripts", json("POST", { ...payload, id })), id);
      await remember(id, { name: created.name, cloud_version: null, base_code: created.code!, local_version: created.version });
      return created;
    });
    return synchronize(local);
  }
  async function saveLocal(id: string, body: string): Promise<ScriptFile> {
    const payload = JSON.parse(body) as { code: string; version: number };
    if (!payload || typeof payload.code !== "string" || Array.from(payload.code).length > 64000
        || !Number.isSafeInteger(payload.version) || payload.version < 0
        || Object.keys(payload).some((key) => !["code", "version"].includes(key)))
      throw new Error("Invalid Python file.");
    return localOperation(async () => {
      editable();
      const entry = current().scripts?.[id];
      if (!entry) throw new Error("The file's sync record could not be found. Reopen the file.");
      if (entry.conflict) throw conflict();
      const local = document(await deps.local.request<ScriptFile>(scriptFilePath(id), json("PUT", payload)), id);
      // The immutable cloud base is already durable. This local CAS document is
      // enough to recover and retry even if the app exits before an upload.
      return decorated(local);
    });
  }
  async function syncId(id: string): Promise<ScriptFile | null> {
    const local = await readLocal(id);
    if (!local) return null;
    return synchronize(local);
  }
  async function retry() {
    if (!deps.state() || deps.state()?.access_denied || deps.state()?.role === "viewer") return;
    for (const [id, entry] of Object.entries(current().scripts ?? {})) {
      if (entry.conflict) continue;
      const local = await localFile(id);
      if (!local || (entry.cloud_version !== null && local.code === entry.base_code)) continue;
      try { if ((await synchronize(local)).pending_sync) return; }
      catch (error) {
        if (!(error instanceof ApiError && error.status === 409)) throw error;
      }
    }
  }
  return { catalog, read, readLocal, create, saveLocal, syncId, retry };
}
