/** One local worker per project; cloud only stores drafts, files and results. */
import {
  ApiError, createWorkspaceClient, scriptFilePath, type WorkspaceClient, type WorkspaceBootstrap,
  validateScriptFile, type ScriptDraft, type ScriptFile, type ScriptSyncNotification,
  type TokenGetter, type WorkspaceSession, type ScriptConflictSnapshot,
} from "./api.ts";
import { nativeInvoke, desktopCloudRequest, isTransientDesktopFailure, type CachedCloudFile } from "./desktop.ts";
import type { DatasetProfile, ConsoleState, ExecutionRecord } from "./types.ts";
import type { PackageSnapshot, PackageManifest } from "./package-types.ts";
import { createNamedScriptSync, type DesktopSyncState, type SyncStateUpdate } from "./desktop-script-sync.ts";
import { createFileLayoutSync } from "./file-layout-sync.ts";
import type { FileLayout } from "./file-layout.ts";
import { createResultSharing } from "./result-sharing.ts";

type SyncState = DesktopSyncState;
type LocalDataset = DatasetProfile & { local_only?: boolean };
interface Dependencies {
  actorUid: string;
  local: WorkspaceClient;
  cloud<T>(path: string, options?: RequestInit): Promise<T>;
  cacheFile(file: DatasetProfile): Promise<CachedCloudFile>;
  cacheFiles?(files: DatasetProfile[]): Promise<CachedCloudFile[]>;
  chooseAndUpload?: () => Promise<DatasetProfile | null>;
}
const recoverable = isTransientDesktopFailure;
function canonicalManifest(value: unknown): string {
  function ordered(item: unknown): unknown {
    if (Array.isArray(item)) return item.map(ordered);
    if (item && typeof item === "object")
      return Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b)).map(([key, nested]) => [key, ordered(nested)]));
    return item;
  }
  return JSON.stringify(ordered(value));
}

/** Cloud archives can contain a summary, never a workspace-local graph URL. */
export function cloudArchiveRecord(record: ExecutionRecord): { record: ExecutionRecord; local_plot_count: number } {
  const publicRecord = { ...record } as ExecutionRecord & Record<string, unknown>;
  for (const key of ["sharing_context", "sharing", "local_only", "sharing_state", "sharing_error", "sharing_retryable", "sharing_metadata"])
    delete publicRecord[key];
  let localPlotCount = 0;
  const outputs = record.outputs.map((output) => {
    const data = output.data;
    if (output.type !== "plot" || !data || typeof data !== "object" ||
        (data as { kind?: unknown }).kind !== "network" ||
        !Object.prototype.hasOwnProperty.call(data, "artifact")) return output;
    localPlotCount++;
    const title = (data as { title?: unknown }).title;
    const name = typeof title === "string" && title.trim() ? title.slice(0, 256) : "Network chart";
    // Keep output indexes, ordered events and publication-ready LaTeX intact.
    // The complete plot and its reference remain in the local execution record.
    return { ...output, type: "text" as const,
      data: `${name}: the complete network chart is stored on the computer where this run was executed. ` +
        "The chart was not uploaded. Export and share its HTML file to share the complete view." };
  });
  return { record: localPlotCount ? { ...publicRecord, outputs } : publicRecord, local_plot_count: localPlotCount };
}

export function createHybridWorkspace(projectId: string, role: SyncState["role"], deps: Dependencies): WorkspaceClient {
  let state: SyncState | null = null;
  let session: WorkspaceSession | null = null;
  let conflict = false;
  let inputFiles: { id: string; data_hash: string }[] = [];
  let onlineListener = false;
  let serial: Promise<unknown> = Promise.resolve();
  let sharedEnvironment: { manifest: PackageManifest | null; version: number } | null = null;
  let environmentWrites: Promise<unknown> = Promise.resolve();
  let closed = false;
  let authorizationDenied = false;
  let denialGeneration = 0;
  let layoutSyncScheduled = false;
  let layoutSyncDirty = false;
  let stateWrites: Promise<unknown> = Promise.resolve();
  let mainWrites: Promise<unknown> = Promise.resolve();
  let scriptSyncScheduled = false;
  const scriptSyncDirty = new Set<string>();
  let scriptRefreshScheduled = false;
  const scriptRefreshDirty = new Set<string>();
  const layoutListeners = new Set<(layout: FileLayout) => void>();
  const scriptListeners = new Set<(value: ScriptSyncNotification) => void>();
  const local = deps.local;
  const syncState = () => authorizationDenied && state ? { ...state, access_denied: true } : state;
  const scripts = createNamedScriptSync({ local, cloud: deps.cloud, state: syncState, persist: persistState, closed: () => closed });
  const fileLayout = createFileLayoutSync({ local, cloud: deps.cloud, state: syncState, persist: persistState });
  function accessible(editing = false) {
    if (closed) throw new DOMException("Workspace closed", "AbortError");
    if (!state) throw new Error("Open the project first.");
    if (authorizationDenied || state.access_denied || (editing && state.role === "viewer"))
      throw new ApiError("You do not have permission to edit this project.", 403, "ROLE_REQUIRED");
    return state;
  }
  const resultSharing = createResultSharing({
    local, cloud: deps.cloud, actorUid: deps.actorUid, accessible, closed: () => closed,
    archive: cloudArchiveRecord,
    deny: async () => { await persistState((latest) => ({ ...latest, access_denied: true })); },
  });
  function resumeResultSharing() {
    if (closed) return;
    if (state?.role === "viewer") {
      void resultSharing.refresh().catch(() => {}); // Viewers may read, never publish.
      return;
    }
    resultSharing.start();
  }
  function mainOperation<T>(operation: () => Promise<T>): Promise<T> {
    const next = mainWrites.catch(() => {}).then(() => {
      accessible();
      return operation();
    });
    mainWrites = next;
    return next;
  }
  async function readMainLocal(): Promise<ScriptFile> {
    return mainOperation(async () => {
      const draft = validateScriptFile(await local.request<ScriptFile>("/console/scripts/analysis"), "analysis");
      const saved = accessible();
      return { ...draft, pending_sync: (draft.code ?? "") !== saved.base_code || conflict,
        ...(conflict ? { sync_conflict: true } : {}) };
    });
  }
  async function notifyScript(id: string, error?: unknown, sourceChanged = false) {
    if (closed || !scriptListeners.size) return;
    const value: ScriptSyncNotification = {};
    if (authorizationDenied || state?.access_denied || state?.role === "viewer") value.read_only = true;
    try { value.file = id === "analysis" ? await readMainLocal() : (await scripts.readLocal(id)) ?? undefined; }
    catch (failure) {
      // A revoked project cannot be read through the normal cached API.
      if (!authorizationDenied && !state?.access_denied) error ??= failure;
    }
    if (closed) return;
    value.sync_conflict = id === "analysis" ? conflict : Boolean(state?.scripts?.[id]?.conflict);
    if (sourceChanged) value.source_changed = true;
    if (error && !recoverable(error) && !(error instanceof DOMException && error.name === "AbortError")) {
      const failure = error instanceof ApiError ? error
        : new ApiError(error instanceof Error ? error.message : "Could not sync the Python file.", 0, "SYNC_FAILED");
      value.sync_error = { message: failure.message, status: failure.status, code: failure.code };
    }
    for (const listener of scriptListeners) {
      try { listener(structuredClone(value)); }
      catch { /* A detached editor cannot fail durable synchronization. */ }
    }
  }
  function queueScriptSync(id: string) {
    if (closed) return;
    scriptSyncDirty.add(id);
    if (scriptSyncScheduled) return;
    scriptSyncScheduled = true;
    const next = serial.catch(() => {}).then(async () => {
      try {
        const batch = [...scriptSyncDirty];
        scriptSyncDirty.clear();
        for (const currentId of batch) {
          if (closed || state?.access_denied) break;
          try {
            if (currentId === "analysis") await synchronizeMain(await readMainLocal());
            else await scripts.syncId(currentId);
            await notifyScript(currentId);
          } catch (error) {
            await notifyScript(currentId, error);
          }
        }
      } finally {
        scriptSyncScheduled = false;
        // New edits are coalesced, and an offline failure is retried only on a
        // later edit/reconnection rather than an unbounded busy loop.
        if (scriptSyncDirty.size && !closed && !state?.access_denied)
          queueScriptSync(scriptSyncDirty.values().next().value!);
      }
    });
    serial = next;
    void next.catch(() => {});
  }
  function queueScriptRefresh(id: string) {
    if (closed) return;
    scriptRefreshDirty.add(id);
    if (scriptRefreshScheduled) return;
    scriptRefreshScheduled = true;
    const next = serial.catch(() => {}).then(async () => {
      try {
        const batch = [...scriptRefreshDirty];
        scriptRefreshDirty.clear();
        for (const currentId of batch) {
          if (closed || state?.access_denied) break;
          let sourceChanged = false;
          try {
            if (currentId === "analysis") sourceChanged = await refreshMain();
            else await scripts.read(currentId, false, () => { sourceChanged = true; });
            await notifyScript(currentId, undefined, sourceChanged);
          } catch (error) {
            await notifyScript(currentId, error);
          }
        }
      } finally {
        scriptRefreshScheduled = false;
        if (scriptRefreshDirty.size && !closed && !state?.access_denied)
          queueScriptRefresh(scriptRefreshDirty.values().next().value!);
      }
    });
    serial = next;
    void next.catch(() => {});
  }
  async function notifyLayout(error?: unknown) {
    if (closed) return;
    const latest = await fileLayout.readLocal(true);
    if (closed) return;
    if (error && !recoverable(error)) {
      const failure = error instanceof ApiError ? error
        : new ApiError(error instanceof Error ? error.message : "Could not sync the file layout.", 0, "SYNC_FAILED");
      latest.sync_error = { message: failure.message, status: failure.status, code: failure.code };
    }
    for (const listener of layoutListeners) {
      try { listener(structuredClone(latest)); }
      catch { /* A detached UI must never fail the durable sync queue. */ }
    }
  }
  function queueLayoutSync() {
    if (closed) return;
    layoutSyncDirty = true;
    if (layoutSyncScheduled) return;
    layoutSyncScheduled = true;
    const next = serial.catch(() => {}).then(async () => {
      try {
        layoutSyncDirty = false;
        if (closed) return;
        // Newly created scripts must exist remotely before layout inventory CAS.
        await scripts.retry();
        if (closed) return;
        await fileLayout.retry();
        await notifyLayout();
      } catch (error) {
        try { await notifyLayout(error); }
        catch { /* Reopening also reads the durable pending/conflict metadata. */ }
      } finally {
        layoutSyncScheduled = false;
        // Append another pass behind already queued source saves. Rapid renames
        // coalesce without keeping the shared remote queue indefinitely busy.
        if (layoutSyncDirty && !closed && !state?.access_denied) queueLayoutSync();
      }
    });
    serial = next;
    // Background errors are delivered through the guarded layout subscription.
    void next.catch(() => {});
  }
  const retryOnline = () => {
    const next = serial.catch(() => {}).then(async () => {
      try {
        if (closed) return;
        await ready();
        const draft = await local.request<ScriptDraft>("/console/script");
        if (state && draft.code !== state.base_code && !conflict && state.role !== "viewer")
          try {
            await synchronizeMain({ ...draft, id: "analysis", version: 0 }, false);
            await notifyScript("analysis");
          } catch (error) {
            await notifyScript("analysis", error);
            if (!(error instanceof ApiError && error.status === 409)) throw error;
          }
        await scripts.retry();
        await fileLayout.retry();
        await notifyLayout();
      } catch (error) {
        try { await notifyLayout(error); }
        catch { /* Preserve the original retry error if local refresh fails. */ }
        throw error;
      }
    });
    serial = next;
    // Result delivery has an independent queue: a source conflict must not
    // strand results, while permission and close guards remain authoritative.
    void next.finally(resumeResultSharing).catch(() => {
      // Source/layout errors use their own subscriptions. Delivery reports its
      // independent failures and must not inherit a source-version conflict.
    });
  };
  async function persistState(update: SyncStateUpdate, clearDenied = false) {
    // Object callers predate local/background concurrency. Capture their actual
    // field changes now, then merge only those fields into the latest state.
    const previous = state;
    const intended = typeof update === "function" ? (state ? update(state) : null) : update;
    if (intended?.access_denied) { authorizationDenied = true; denialGeneration += 1; }
    const generation = denialGeneration;
    const changes = typeof update === "function" ? null
      : Object.fromEntries(Object.entries(update).filter(([key, value]) =>
        canonicalManifest(value) !== canonicalManifest(previous?.[key as keyof SyncState])));
    const next = stateWrites.catch(() => {}).then(async () => {
      if (closed) throw new DOMException("Workspace closed", "AbortError");
      if (typeof update === "function" && !state) throw new Error("Open the project first.");
      const nextState = typeof update === "function"
        ? update(state!)
        : { ...state, ...changes } as SyncState;
      await local.request("/desktop-sync-state", { method: "PUT", body: JSON.stringify(nextState) });
      if (closed) return;
      state = nextState;
      if (nextState.access_denied) authorizationDenied = true;
      else if (clearDenied && generation === denialGeneration) authorizationDenied = false;
    });
    stateWrites = next;
    await next;
  }
  async function ready() {
    if (closed) throw new DOMException("Workspace closed", "AbortError");
    if (!onlineListener && typeof window !== "undefined") {
      window.addEventListener("online", retryOnline); onlineListener = true;
    }
    if (!session) session = await local.connect();
    if (!state) state = await local.request<SyncState | null>("/desktop-sync-state");
    return session;
  }
  async function importFile(file: DatasetProfile, prefetched?: CachedCloudFile) {
    if ((file as LocalDataset).local_only === true) {
      // Native imports are already owned by this local project. Never send
      // their UUIDs through the cloud cache/download path.
      if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(file.id))
        throw new Error("Could not verify the local dataset identifier.");
      const saved = await local.request<LocalDataset>(`/datasets/${file.id}`);
      if (saved.local_only !== true || saved.id !== file.id || saved.data_hash !== file.data_hash)
        throw new Error("Could not verify the local-only dataset record.");
      return saved;
    }
    const cached = prefetched ?? await deps.cacheFile(file);
    if (cached.sha256 !== file.data_hash || cached.cloud_id !== file.id)
      throw new Error("Could not verify the downloaded data version.");
    const imported = await local.request<DatasetProfile>("/datasets/import-cached", {
      method: "POST", body: JSON.stringify({ name: cached.name, expected_sha256: cached.sha256, cloud_id: file.id }),
    });
    inputFiles = [...inputFiles.filter((input) => input.id !== file.id), { id: file.id, data_hash: file.data_hash }];
    return imported;
  }
  async function cachedBootstrap(localSession: WorkspaceSession): Promise<WorkspaceBootstrap> {
      if (!state) throw new Error("The project has not been cached yet.");
      const [console, datasets, cached, localDraft] = await Promise.all([
        local.request<ConsoleState>("/console"), local.request<{ datasets: DatasetProfile[] }>("/datasets"),
        local.request<{ files: { cloud_id: string; sha256: string }[] }>("/desktop-cached-files"),
        local.request<ScriptDraft>("/console/script"),
      ]);
      inputFiles = cached.files.map((file) => ({ id: file.cloud_id, data_hash: file.sha256 }));
      return {
        session: { ...localSession, read_only: state.role === "viewer", execution_mode: "persistent" },
        draft: { ...localDraft, version: state.cloud_version, pending_sync: localDraft.code !== state.base_code },
        datasets: datasets.datasets, status: console.status, initialConsole: console,
      };
  }
  async function bootstrap(): Promise<WorkspaceBootstrap> {
    const localSession = await ready();
    const hadCachedState = state !== null;
    const localDraft = await local.request<ScriptDraft>("/console/script");
    let remote: WorkspaceBootstrap;
    try {
      remote = await deps.cloud<WorkspaceBootstrap>("/bootstrap");
    } catch (error) {
      if (state && error instanceof ApiError && (error.status === 401 || error.status === 403))
        await persistState({ ...state, access_denied: true });
      if (!hadCachedState || !recoverable(error)) throw error;
      if (state?.access_denied) throw new ApiError("Could not verify your access to the project. Sign in again.", 403, "ROLE_REQUIRED");
      const cached = await cachedBootstrap(localSession);
      void resultSharing.refresh().catch(() => {}); // refresh reports metadata failures.
      return cached;
    }
    if (!Number.isInteger(remote.draft.version)) throw new Error("Could not verify the cloud draft version.");
    const remoteCode = remote.draft.code ?? "";
    const localCode = localDraft.code ?? "";
    const effectiveRole: SyncState["role"] = remote.session.read_only ? "viewer" : role === "viewer" ? "editor" : role;
    const pending = state !== null ? localCode !== state.base_code : Boolean(localCode && localCode !== remoteCode);
    if (!state && pending) {
      conflict = true;
      await persistState({ cloud_version: remote.draft.version!, base_code: remoteCode, role: effectiveRole });
    }
    if (pending && remote.draft.version !== state!.cloud_version) {
      conflict = true;
      // Preserve the exact local draft. The UI's existing conflict flow can export it.
    } else if (!pending) {
      await local.request("/console/script", { method: "PUT", body: JSON.stringify({ code: remoteCode, name: "analysis.py" }) });
      await persistState({ ...state, cloud_version: remote.draft.version!, base_code: remoteCode, role: effectiveRole });
    }
    if (state!.role !== effectiveRole || state!.access_denied || authorizationDenied)
      await persistState({ ...state!, role: effectiveRole, access_denied: false }, true);
    // Bounded sequential downloads avoid multiplying the full dataset working set.
    const datasets: DatasetProfile[] = [];
    try {
      const batch = deps.cacheFiles ? await deps.cacheFiles(remote.datasets) : null;
      for (const file of remote.datasets)
        datasets.push(await importFile(file, batch?.find((cached) => cached.cloud_id === file.id)));
    } catch (error) {
      if (!hadCachedState || !recoverable(error)) throw error;
      return cachedBootstrap(localSession);
    }
    inputFiles = remote.datasets.map((file) => ({ id: file.id, data_hash: file.data_hash }));
    const persisted = await local.request<{ datasets: LocalDataset[] }>("/datasets");
    const localOnly = persisted.datasets.filter((file) => file.local_only === true);
    datasets.push(...localOnly.filter((file) => !datasets.some((shared) => shared.id === file.id)));
    const console = await local.request<ConsoleState>("/console");
    // A reopened desktop may already be online, so no browser online event will
    // arrive to flush files created during the previous offline session.
    await scripts.retry();
    resumeResultSharing(); // Reopened online projects also resume durable deliveries.
    return {
      session: { ...localSession, read_only: remote.session.read_only, execution_mode: "persistent" },
      draft: { code: pending ? localCode : remote.draft.code, name: "analysis.py", version: state!.cloud_version,
               pending_sync: pending },
      datasets, status: console.status, initialConsole: console,
    };
  }
  async function saveDraft(body: string): Promise<ScriptDraft> {
    await ready();
    const draft = JSON.parse(body) as ScriptDraft;
    if (typeof draft.code !== "string") throw new Error("Invalid draft.");
    if (state?.role === "viewer" || state?.access_denied)
      throw new ApiError("You do not have permission to edit this project.", 403, "ROLE_REQUIRED");
    await local.request("/console/script", { method: "PUT", body: JSON.stringify({ code: draft.code, name: "analysis.py" }) });
    if (!state) throw new Error("Project sync has not started; your draft is preserved locally.");
    if (conflict) throw new ApiError("The cloud draft has changed. Download your local code and open the current version.", 409, "VERSION_CONFLICT");
    try {
      const remote = await deps.cloud<ScriptDraft>("/console/script", {
        method: "PUT", body: JSON.stringify({ code: draft.code, version: state.cloud_version }),
      });
      if (!Number.isInteger(remote.version)) throw new Error("Could not verify the cloud record.");
      await persistState({ ...state, cloud_version: remote.version!, base_code: draft.code });
      return { ...remote, pending_sync: false };
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) conflict = true;
      if (error instanceof ApiError && (error.status === 401 || error.status === 403))
        await persistState({ ...state, access_denied: true });
      if (!recoverable(error)) throw error;
      return { code: draft.code, name: "analysis.py", version: state.cloud_version, pending_sync: true };
    }
  }
  async function synchronizeMain(draft: ScriptFile, readback = true): Promise<ScriptFile> {
    const saved = accessible(true);
    if (conflict) throw new ApiError("The cloud draft has changed. Your local code is preserved.", 409, "VERSION_CONFLICT");
    const code = draft.code ?? "";
    if (code === saved.base_code) return readback ? readMainLocal() : draft;
    try {
      const remote = await deps.cloud<ScriptDraft>("/console/script", {
        method: "PUT", body: JSON.stringify({ code, version: saved.cloud_version }),
      });
      accessible();
      if (!Number.isSafeInteger(remote.version) || remote.version! <= saved.cloud_version || remote.code !== code)
        throw new Error("Could not verify the cloud record.");
      await persistState((latest) => ({ ...latest, cloud_version: remote.version!, base_code: code }));
    } catch (error) {
      if (closed) throw new DOMException("Workspace closed", "AbortError");
      if (error instanceof ApiError && (error.status === 401 || error.status === 403))
        await persistState((latest) => ({ ...latest, access_denied: true }));
      if (error instanceof ApiError && error.status === 409) {
        // A lost acknowledgement can advance the remote version without losing
        // our text. Exact readback settles that case; peers are never overwritten.
        let remote: ScriptDraft;
        try { remote = await deps.cloud<ScriptDraft>("/console/script"); }
        catch (failure) {
          if (failure instanceof ApiError && (failure.status === 401 || failure.status === 403))
            await persistState((latest) => ({ ...latest, access_denied: true }));
          throw failure;
        }
        accessible();
        if (remote.code === code && Number.isSafeInteger(remote.version))
          await persistState((latest) => ({ ...latest, cloud_version: remote.version!, base_code: code }));
        else {
          conflict = true;
          throw new ApiError("The cloud draft has changed. Your local code is preserved.", 409, "VERSION_CONFLICT");
        }
      } else if (!recoverable(error)) throw error;
    }
    // A newer local edit may have landed while the acknowledgement was in
    // flight. Only the acknowledged base advances; this fresh snapshot stays pending.
    return readback ? readMainLocal() : draft;
  }
  async function refreshMain(): Promise<boolean> {
    accessible();
    let remote: ScriptDraft;
    try { remote = await deps.cloud<ScriptDraft>("/console/script"); }
    catch (error) {
      if (!closed && state && error instanceof ApiError && (error.status === 401 || error.status === 403))
        await persistState((latest) => ({ ...latest, access_denied: true }));
      throw error;
    }
    if (!Number.isSafeInteger(remote.version) || remote.version! < 0
        || (remote.code !== null && (typeof remote.code !== "string" || Array.from(remote.code).length > 64000)))
      throw new Error("Could not verify the cloud draft.");
    return mainOperation(async () => {
      const saved = accessible();
      const latest = validateScriptFile(await local.request<ScriptFile>("/console/scripts/analysis"), "analysis");
      const remoteCode = remote.code ?? "";
      const localCode = latest.code ?? "";
      if (conflict) return false;
      if (localCode !== saved.base_code && remoteCode !== localCode && remoteCode !== saved.base_code) {
        conflict = true;
        return false;
      }
      if (localCode !== saved.base_code && remoteCode === saved.base_code) {
        if (remote.version !== saved.cloud_version)
          await persistState((current) => ({ ...current, cloud_version: remote.version! }));
        return false;
      }
      let sourceChanged = false;
      if (localCode !== remoteCode) {
        const imported = validateScriptFile(await local.request<ScriptFile>("/desktop-scripts/analysis", {
          method: "PUT", body: JSON.stringify({ name: "analysis.py", code: remoteCode, version: latest.version }),
        }), "analysis");
        sourceChanged = imported.code !== latest.code;
      }
      if (remote.version !== saved.cloud_version || remoteCode !== saved.base_code)
        await persistState((current) => ({ ...current, cloud_version: remote.version!, base_code: remoteCode }));
      return sourceChanged;
    });
  }
  async function saveMainLocal(body: string): Promise<ScriptFile> {
    const payload = JSON.parse(body) as { code: string; version: number };
    if (!payload || typeof payload.code !== "string" || Array.from(payload.code).length > 64000
        || !Number.isSafeInteger(payload.version) || payload.version < 0
        || Object.keys(payload).some((key) => !["code", "version"].includes(key)))
      throw new Error("Invalid Python file.");
    return mainOperation(async () => {
      const saved = accessible(true);
      if (conflict) throw new ApiError("The cloud draft has changed. Your local code is preserved.", 409, "VERSION_CONFLICT");
      const draft = validateScriptFile(await local.request<ScriptFile>("/console/scripts/analysis", {
        method: "PUT", body: JSON.stringify(payload),
      }), "analysis");
      accessible(true);
      return { ...draft, pending_sync: (draft.code ?? "") !== saved.base_code };
    });
  }
  async function conflictCloud<T>(
    path: string,
    options: RequestInit = {},
  ): Promise<T> {
    try {
      return await deps.cloud<T>(path, options);
    } catch (error) {
      if (
        error instanceof ApiError &&
        (error.status === 401 || error.status === 403) &&
        !closed
      )
        await persistState((current) => ({ ...current, access_denied: true }));
      throw error;
    }
  }
  return {
    projectId, teams: true, desktop: true,
    connect: ready,
    readScriptConflict: async (id: string): Promise<ScriptConflictSnapshot> => {
      accessible();
      scriptFilePath(id);
      const cached = validateScriptFile(
        await local.request<ScriptFile>(`/console/scripts/${id}`),
        id,
      );
      const remotePath =
        id === "analysis" ? "/console/script" : `/console/scripts/${id}`;
      const rawRemote = await conflictCloud<ScriptFile>(remotePath);
      const remote = validateScriptFile(
        id === "analysis" ? { ...rawRemote, id } : rawRemote,
        id,
      );
      const current = accessible();
      return {
        local: cached,
        remote,
        base:
          id === "analysis"
            ? current.base_code
            : (current.scripts?.[id]?.base_code ?? ""),
      };
    },
    resolveScriptConflict: async (snapshot, code) => {
      const next = serial
        .catch(() => {})
        .then(async () => {
          accessible(true);
          const id = snapshot.local.id;
          scriptFilePath(id);
          if (
            snapshot.remote.id !== id ||
            typeof code !== "string" ||
            Array.from(code).length > 64000
          )
            throw new Error("Invalid merged draft.");
          const latest = validateScriptFile(
            await local.request<ScriptFile>(`/console/scripts/${id}`),
            id,
          );
          if (
            latest.version !== snapshot.local.version ||
            latest.code !== snapshot.local.code
          )
            throw new ApiError(
              "The local file changed again. Compare the latest versions.",
              409,
              "VERSION_CONFLICT",
            );
          const remotePath =
            id === "analysis" ? "/console/script" : `/console/scripts/${id}`;
          const rawRemote = await conflictCloud<ScriptFile>(remotePath, {
            method: "PUT",
            body: JSON.stringify({
              code,
              version: snapshot.remote.version,
              ...(id === "analysis" ? { name: "analysis.py" } : {}),
            }),
          });
          const remote = validateScriptFile(
            id === "analysis" ? { ...rawRemote, id } : rawRemote,
            id,
          );
          accessible(true);
          if (remote.code !== code)
            throw new Error("Could not verify the merged cloud draft.");
          const saved = validateScriptFile(
            await local.request<ScriptFile>(`/desktop-scripts/${id}`, {
              method: "PUT",
              body: JSON.stringify({
                name: remote.name,
                code,
                version: latest.version,
              }),
            }),
            id,
          );
          await persistState((current) =>
            id === "analysis"
              ? { ...current, cloud_version: remote.version, base_code: code }
              : {
                  ...current,
                  scripts: {
                    ...current.scripts,
                    [id]: {
                      name: remote.name,
                      cloud_version: remote.version,
                      base_code: code,
                      local_version: saved.version,
                      conflict: false,
                    },
                  },
                },
          );
          if (id === "analysis") conflict = false;
          return { ...saved, pending_sync: false, sync_conflict: false };
        });
      serial = next;
      return next;
    },
    cancelPending: () => {
      closed = true;
      layoutSyncDirty = false;
      scriptSyncDirty.clear();
      scriptRefreshDirty.clear();
      layoutListeners.clear();
      scriptListeners.clear();
      resultSharing.cancel();
      fileLayout.cancel();
      local.cancelPending();
      if (onlineListener && typeof window !== "undefined") window.removeEventListener("online", retryOnline);
      onlineListener = false;
    },
    subscribeFileLayout: (listener) => {
      if (closed) return () => {};
      layoutListeners.add(listener);
      return () => { layoutListeners.delete(listener); };
    },
    subscribeScriptSync: (listener) => {
      if (closed) return () => {};
      scriptListeners.add(listener);
      return () => { scriptListeners.delete(listener); };
    },
    subscribeResultSharing: resultSharing.subscribe,
    async request<T>(path: string, options: RequestInit = {}, plainText: boolean | "blob" = false): Promise<T> {
      const executionInputs = path === "/console/execute" && options.method === "POST"
        ? structuredClone(inputFiles) : null;
      if (path === "/bootstrap") {
        const next = serial.catch(() => {}).then(bootstrap);
        serial = next;
        return next as Promise<T>;
      }
      await ready();
      if (authorizationDenied || state?.access_denied) throw new ApiError("Could not verify your access to the project. Reopen the project.", 403, "ROLE_REQUIRED");
      if (path === "/result-sharing" && (options.method ?? "GET") === "GET")
        return resultSharing.refresh() as Promise<T>;
      const resultRetry = /^\/result-sharing\/([^/]+)\/retry$/.exec(path);
      if (resultRetry && options.method === "POST")
        return resultSharing.retry(decodeURIComponent(resultRetry[1])) as Promise<T>;
      if (path === "/console/history" || path.startsWith("/console/history?") || /^\/runs\/[A-Za-z0-9_-]+\/record$/.test(path)) {
        // Browsing a shared archive never executes Python or replaces local history.
        if (options.method && options.method !== "GET") throw new ApiError("Unsupported history operation.", 405);
        return deps.cloud<T>(path, options);
      }
      if (path === "/files/layout") {
        const method = options.method ?? "GET";
        if (method !== "GET" && method !== "PUT") throw new ApiError("Unsupported file layout operation.", 405);
        if (method === "PUT") {
          // Logical renames are durable local metadata operations. They do not
          // wait for the executable-source queue or a cloud catalogue refresh.
          const saved = await fileLayout.saveLocal(String(options.body ?? "{}"));
          if (saved.pending_sync) queueLayoutSync();
          return saved as T;
        }
        if (new Headers(options.headers).get("X-OpenEcon-Local-Only") === "true")
          return fileLayout.readLocal() as Promise<T>;
        const next = serial.catch(() => {}).then(async () => {
          if (state?.access_denied) throw new ApiError("Could not verify your access to the project. Reopen the project.", 403, "ROLE_REQUIRED");
          // The persisted metadata catalog includes unopened cloud scripts.
          await scripts.catalog();
          const resolveConflict = new Headers(options.headers).get("X-OpenEcon-Resolve-Conflict") === "remote";
          return fileLayout.read(resolveConflict);
        });
        serial = next;
        return next as Promise<T>;
      }
      if (/^\/console\/scripts\/(?:analysis|[0-9a-f]{32})$/.test(path)) {
        const method = options.method ?? "GET";
        const id = path.split("/").at(-1)!;
        const resolveConflict = new Headers(options.headers).get("X-OpenEcon-Resolve-Conflict") === "remote";
        if (method === "GET" && !resolveConflict) {
          const cached = id === "analysis" ? await readMainLocal() : await scripts.readLocal(id);
          if (cached) {
            queueScriptRefresh(id);
            return cached as T;
          }
        }
        if (method === "PUT") {
          const saved = id === "analysis" ? await saveMainLocal(String(options.body ?? "{}"))
            : await scripts.saveLocal(id, String(options.body ?? "{}"));
          if (saved.pending_sync) queueScriptSync(id);
          return saved as T;
        }
        if (method !== "GET") throw new ApiError("Unsupported file operation.", 405);
        const next = serial.catch(() => {}).then(async () => {
          accessible();
          if (id !== "analysis") return scripts.read(id, resolveConflict);
          const previous = await readMainLocal();
          let remote: ScriptDraft;
          try { remote = await deps.cloud<ScriptDraft>("/console/script"); }
          catch (error) {
            if (state && error instanceof ApiError && (error.status === 401 || error.status === 403))
              await persistState((latest) => ({ ...latest, access_denied: true }));
            throw error;
          }
          accessible();
          if (!Number.isSafeInteger(remote.version)) throw new Error("Could not verify the cloud draft version.");
          const imported = await mainOperation(async () => {
            const draft = validateScriptFile(await local.request<ScriptFile>("/desktop-scripts/analysis", {
              method: "PUT", body: JSON.stringify({ name: "analysis.py", code: remote.code ?? "", version: previous.version }),
            }), "analysis");
            await persistState((latest) => ({ ...latest, cloud_version: remote.version!, base_code: remote.code ?? "" }));
            conflict = false;
            return { ...draft, pending_sync: false };
          });
          return imported;
        });
        serial = next;
        return next as Promise<T>;
      }
      if (path === "/console/scripts") {
        const next = serial.catch(() => {}).then(async () => {
          if (state?.access_denied) throw new ApiError("Could not verify your access to the project. Reopen the project.", 403, "ROLE_REQUIRED");
          const method = options.method ?? "GET";
          if (method === "GET") return scripts.catalog();
          if (method === "POST") return scripts.create(String(options.body ?? "{}"));
          throw new ApiError("Unsupported file operation.", 405);
        });
        serial = next;
        return next as Promise<T>;
      }
      if (path === "/environment/shared") {
        try {
          sharedEnvironment = await deps.cloud<typeof sharedEnvironment>("/environment");
          if (!sharedEnvironment || !Number.isInteger(sharedEnvironment.version))
            throw new Error("Could not verify the cloud environment version.");
          const environment = await local.request<PackageSnapshot>("/environment");
          return { ...sharedEnvironment, localMatches: canonicalManifest(environment.manifest) === canonicalManifest(sharedEnvironment.manifest), offline: false } as T;
        } catch (error) {
          if (!recoverable(error)) throw error;
          return { ...(sharedEnvironment ?? { manifest: null, version: 0 }), localMatches: false, offline: true } as T;
        }
      }
      if (path === "/environment/share" && options.method === "POST") {
        const next = environmentWrites.catch(() => {}).then(async () => {
          if (state?.role === "viewer") throw new ApiError("You do not have permission to edit this project.", 403, "ROLE_REQUIRED");
          if (!sharedEnvironment) sharedEnvironment = await deps.cloud("/environment");
          const environment = await local.request<PackageSnapshot>("/environment");
          if (environment.job?.state === "running") throw new ApiError("Wait for the installation to finish.", 409, "ENVIRONMENT_BUSY");
          sharedEnvironment = await deps.cloud("/environment", {
            method: "PUT", body: JSON.stringify({ manifest: environment.manifest, version: sharedEnvironment!.version }),
          });
          return sharedEnvironment;
        });
        environmentWrites = next;
        return next as Promise<T>;
      }
      if (path === "/environment/restore-shared" && options.method === "POST") {
        const latest = await deps.cloud<{ manifest: PackageManifest | null; version: number }>("/environment");
        if (!latest.manifest) throw new Error("No shared environment was found.");
        sharedEnvironment = latest;
        return local.request<T>("/environment/restore", { method: "POST", body: JSON.stringify({ manifest: latest.manifest }) });
      }
      if (path === "/console/script" && options.method === "PUT") {
        const next = serial.catch(() => {}).then(() => saveDraft(String(options.body || "")));
        serial = next;
        return next as Promise<T>;
      }
      if (path === "/console/execute" && options.method === "POST") {
        accessible(true);
        const body = JSON.parse(String(options.body || "{}"));
        delete body.wait_for_result;
        // Capture dependencies before any asynchronous work. The backend saves
        // this exact context and the result before attempting durable enqueue.
        const record = await local.request<ExecutionRecord>("/desktop-console/execute", {
          ...options, body: JSON.stringify({ ...body, actor_uid: deps.actorUid, input_files: executionInputs }),
        });
        accessible(true);
        void resultSharing.refresh(true).catch(() => {}); // Read status after the durable execution write.
        resumeResultSharing();
        const localPlotCount = cloudArchiveRecord(record).local_plot_count;
        return (localPlotCount ? { ...record, sharing_scope: "partial", local_plot_count: localPlotCount } : record) as T;
      }
      if (path === "/datasets/upload" && options.method === "POST") {
        accessible(true);
        if (!deps.chooseAndUpload) throw new Error("File selection is unavailable.");
        const uploaded = await deps.chooseAndUpload();
        accessible(true);
        if (!uploaded) throw new DOMException("File selection cancelled", "AbortError");
        return importFile(uploaded) as Promise<T>;
      }
      if (path === "/datasets/example" && options.method === "POST") {
        const file = await deps.cloud<DatasetProfile>(path, options);
        return importFile(file) as Promise<T>;
      }
      // Execution and stop/reset/history/data access ALWAYS remain local.
      return local.request<T>(path, options, plainText);
    },
  };
}

export function createDesktopWorkspaceClient(projectId: string, getToken: TokenGetter,
                                              role: SyncState["role"], actorUid: string): WorkspaceClient {
  const local = createWorkspaceClient(null, undefined, `/api/desktop/projects/${encodeURIComponent(projectId)}/workspace`);
  const cloudPrefix = `/api/projects/${encodeURIComponent(projectId)}/workspace`;
  const cloud = async <T>(path: string, options: RequestInit = {}): Promise<T> => {
    try { return await desktopCloudRequest<T>(cloudPrefix + path, await getToken(), options); }
    catch (error) {
      if (!(error instanceof ApiError) || error.status !== 401) throw error;
      return desktopCloudRequest<T>(cloudPrefix + path, await getToken(true), options);
    }
  };
  return createHybridWorkspace(projectId, role, {
    actorUid, local, cloud,
    cacheFile: async (file) => nativeInvoke<CachedCloudFile>("download_project_file", {
      projectId, fileId: file.id, token: await getToken(),
    }),
    cacheFiles: async () => nativeInvoke<CachedCloudFile[]>("download_project_files", {
      projectId, token: await getToken(),
    }),
    chooseAndUpload: async () => {
      const response = await nativeInvoke<{ status: number; body: DatasetProfile }>("upload_project_file", {
        projectId, token: await getToken(),
      });
      if (response.status === 204) return null;
      if (response.status >= 400) throw new ApiError("Could not upload the data.", response.status);
      return response.body;
    },
  });
}
